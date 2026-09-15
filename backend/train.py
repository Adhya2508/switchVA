import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel

# Set CPU threading
torch.set_num_threads(12)

# Add current workspace to path
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from backend.config import (
    DATASET_PATH,
    PRETRAINED_MODEL_PATH,
    MODEL_SAVE_PATH,
    CHECKPOINT_SAVE_PATH,
    METRICS_SAVE_PATH,
    TEST_METRICS_PATH,
    PREDICTIONS_COMPARISON_PATH,
    SAVE_DIR,
    BATCH_SIZE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    DEVICE,
    MAX_LEN,
    MAX_DISTANCE,
)
from backend.preprocessing import expand_aspect_dataset
from backend.dataset import AspectEmotionDataset, aspect_collate_fn
from backend.model import AspectEmotionRegressor, build_lexicon_features_tensor
from backend.loss import (
    compute_aspect_regression_loss,
    calculate_comprehensive_metrics,
)


def contrastive_pair_loss(pred_val, sample_ids, sentences, targets, margin=0.3):
    """
    For samples from the same sentence that have opposite true polarities
    (one V>0.5, one V<0.5), penalize if their predicted valences are too close.
    This forces the model to produce divergent predictions for contrastive aspects.
    """
    loss = torch.tensor(0.0, device=pred_val.device)
    n = len(sample_ids)
    count = 0
    # Group by sentence prefix (sentence_id without aspect suffix)
    sent_map = {}
    for i, (sid, sent) in enumerate(zip(sample_ids, sentences)):
        key = sent  # group by full sentence text
        if key not in sent_map:
            sent_map[key] = []
        sent_map[key].append(i)

    for indices in sent_map.values():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                ii, jj = indices[i], indices[j]
                tv_i = targets[ii, 0]
                tv_j = targets[jj, 0]
                # Only penalize when true polarities differ
                if (tv_i - 0.5) * (tv_j - 0.5) < 0:
                    pv_i = pred_val[ii]
                    pv_j = pred_val[jj]
                    # Push predicted valences apart by at least `margin`
                    gap = torch.abs(pv_i - pv_j)
                    pair_loss = torch.clamp(margin - gap, min=0.0)
                    loss = loss + pair_loss
                    count += 1

    if count > 0:
        loss = loss / count
    return loss


FEATURE_CACHE_PATH = os.path.join(SAVE_DIR, "cached_aspect_features.pt")


def prepare_aspect_splits(csv_path: str = DATASET_PATH, random_state: int = 42):
    raw_df = pd.read_csv(csv_path)
    aspect_df = expand_aspect_dataset(raw_df)

    unique_sids = aspect_df["sentence_id"].unique()
    train_sids, temp_sids = train_test_split(unique_sids, test_size=0.2, random_state=random_state)
    val_sids, test_sids = train_test_split(temp_sids, test_size=0.5, random_state=random_state)

    train_df = aspect_df[aspect_df["sentence_id"].isin(train_sids)].reset_index(drop=True)
    val_df = aspect_df[aspect_df["sentence_id"].isin(val_sids)].reset_index(drop=True)
    test_df = aspect_df[aspect_df["sentence_id"].isin(test_sids)].reset_index(drop=True)

    print(
        f"Aspect Dataset Splits: Train={len(train_df)} aspects ({len(train_sids)} sents), "
        f"Val={len(val_df)} aspects ({len(val_sids)} sents), "
        f"Test={len(test_df)} aspects ({len(test_sids)} sents)",
        flush=True,
    )

    return train_df, val_df, test_df


def extract_cached_representations(loader, transformer, device):
    feats = []
    switches = []
    targets = []
    sample_ids = []
    sentences = []
    aspects = []
    opinions = []

    with torch.inference_mode():
        for b in loader:
            inp = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            out = transformer(input_ids=inp, attention_mask=mask)
            hidden = out.last_hidden_state  # [B, N, H]

            cls_rep = hidden[:, 0]
            mask_exp = mask.unsqueeze(-1).expand_as(hidden)
            mean_rep = torch.sum(hidden * mask_exp, dim=1) / mask_exp.sum(dim=1).clamp(min=1)
            max_rep = torch.max(hidden + (1.0 - mask_exp) * -1e9, dim=1).values

            joint_rep = torch.cat([cls_rep, mean_rep, max_rep], dim=-1)

            feats.append(joint_rep.cpu())
            switches.append(b["switch_ids"].cpu())
            targets.append(b["target"].cpu())
            sample_ids.extend(b["sample_id"])
            sentences.extend(b["sentence"])
            aspects.extend(b["aspect"])
            opinions.extend(b["opinion"])

    return {
        "features": torch.cat(feats, dim=0),
        "switches": torch.cat(switches, dim=0),
        "targets": torch.cat(targets, dim=0),
        "sample_ids": sample_ids,
        "sentences": sentences,
        "aspects": aspects,
        "opinions": opinions,
    }


def evaluate_feature_regressor(model, data, device):
    model.eval()
    X = data["features"].to(device)
    S = data["switches"].to(device)
    L = data["lexicons"].to(device)
    y = data["targets"].to(device)

    with torch.no_grad():
        outputs = model(X, S, L)
        pred_val = outputs["valence"]
        pred_aro = outputs["arousal"]

        loss_dict = compute_aspect_regression_loss(
            pred_val=pred_val,
            pred_aro=pred_aro,
            target_val=y[:, 0],
            target_aro=y[:, 1],
        )

        pv = pred_val.cpu().numpy()
        pa = pred_aro.cpu().numpy()
        tv = y[:, 0].cpu().numpy()
        ta = y[:, 1].cpu().numpy()

    metrics = calculate_comprehensive_metrics(tv.tolist(), pv.tolist(), ta.tolist(), pa.tolist())
    metrics["loss"] = round(float(loss_dict["loss"].item()), 4)

    records = []
    for i in range(len(pv)):
        records.append({
            "sample_id": data["sample_ids"][i],
            "sentence": data["sentences"][i],
            "aspect": data["aspects"][i],
            "opinion": data["opinions"][i],
            "true_valence": round(float(tv[i]), 4),
            "pred_valence": round(float(pv[i]), 4),
            "valence_error": round(float(abs(pv[i] - tv[i])), 4),
            "true_arousal": round(float(ta[i]), 4),
            "pred_arousal": round(float(pa[i]), 4),
            "arousal_error": round(float(abs(pa[i] - ta[i])), 4),
        })

    return metrics, records


def train_aspect_emotion_regressor(
    epochs: int = 300,
    lr: float = 1.5e-3,
    device: torch.device = DEVICE,
):
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_df, val_df, test_df = prepare_aspect_splits()

    model_source = PRETRAINED_MODEL_PATH if os.path.exists(PRETRAINED_MODEL_PATH) else "l3cube-pune/hing-roberta"
    tokenizer = AutoTokenizer.from_pretrained(model_source)

    # Check if cached representations exist
    if os.path.exists(FEATURE_CACHE_PATH):
        print(f"Loading precomputed contextual representations from {FEATURE_CACHE_PATH}...", flush=True)
        cached_data = torch.load(FEATURE_CACHE_PATH, map_location="cpu")
        train_data = cached_data["train"]
        val_data = cached_data["val"]
        test_data = cached_data["test"]
    else:
        print(f"Extracting contextual representations with HingRoBERTa...", flush=True)
        transformer = AutoModel.from_pretrained(model_source).to(device)
        transformer.eval()

        train_ds = AspectEmotionDataset(train_df, tokenizer=tokenizer, max_len=MAX_LEN)
        val_ds = AspectEmotionDataset(val_df, tokenizer=tokenizer, max_len=MAX_LEN)
        test_ds = AspectEmotionDataset(test_df, tokenizer=tokenizer, max_len=MAX_LEN)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn=aspect_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=aspect_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=aspect_collate_fn)

        train_data = extract_cached_representations(train_loader, transformer, device)
        val_data = extract_cached_representations(val_loader, transformer, device)
        test_data = extract_cached_representations(test_loader, transformer, device)

        torch.save({"train": train_data, "val": val_data, "test": test_data}, FEATURE_CACHE_PATH)
        print(f"Cached contextual representations saved to {FEATURE_CACHE_PATH}", flush=True)

    # Build lexicon prior feature tensors
    train_data["lexicons"] = build_lexicon_features_tensor(train_data["opinions"], train_data["sentences"], train_data["aspects"])
    val_data["lexicons"] = build_lexicon_features_tensor(val_data["opinions"], val_data["sentences"], val_data["aspects"])
    test_data["lexicons"] = build_lexicon_features_tensor(test_data["opinions"], test_data["sentences"], test_data["aspects"])

    # Initialize Regressor
    model = AspectEmotionRegressor(
        emb_dim=train_data["features"].shape[1],
        lex_dim=15,
        hidden_dim=256,
        max_dist=MAX_DISTANCE,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_rmse = float("inf")
    history_records = []

    X_train = train_data["features"].to(device)
    S_train = train_data["switches"].to(device)
    L_train = train_data["lexicons"].to(device)
    y_train = train_data["targets"].to(device)

    print("\n" + "=" * 80, flush=True)
    print(f"STARTING ASPECT-LEVEL EMOTION REGRESSION TRAINING ({epochs} Epochs on {device})", flush=True)
    print("=" * 80, flush=True)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train, S_train, L_train)

        loss_dict = compute_aspect_regression_loss(
            pred_val=outputs["valence"],
            pred_aro=outputs["arousal"],
            target_val=y_train[:, 0],
            target_aro=y_train[:, 1],
        )

        # Contrastive pair penalty — push apart same-sentence opposite-polarity aspects
        c_loss = contrastive_pair_loss(
            outputs["valence"], train_data["sample_ids"], train_data["sentences"],
            y_train, margin=0.30,
        )
        loss = loss_dict["loss"] + 0.4 * c_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        val_metrics, _ = evaluate_feature_regressor(model, val_data, device)

        history_records.append({
            "Epoch": epoch + 1,
            "Train Loss": round(float(loss.item()), 4),
            "Val Loss": val_metrics["loss"],
            "Val Valence RMSE": val_metrics["valence_rmse"],
            "Val Arousal RMSE": val_metrics["arousal_rmse"],
            "Val Overall RMSE": val_metrics["overall_rmse"],
            "Val Valence MAE": val_metrics["valence_mae"],
            "Val Arousal MAE": val_metrics["arousal_mae"],
            "Val Valence R2": val_metrics["valence_r2"],
            "Val Arousal R2": val_metrics["arousal_r2"],
        })

        if (epoch + 1) % 25 == 0 or epoch == 0 or epoch == epochs - 1:
            print(
                f"Epoch [{epoch+1:03d}/{epochs:03d}] | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val-V RMSE: {val_metrics['valence_rmse']:.4f} | "
                f"Val-A RMSE: {val_metrics['arousal_rmse']:.4f} | "
                f"Val-Overall RMSE: {val_metrics['overall_rmse']:.4f} | "
                f"Val-V R²: {val_metrics['valence_r2']:.4f}",
                flush=True,
            )

        if val_metrics["overall_rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["overall_rmse"]
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_rmse": best_val_rmse,
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, CHECKPOINT_SAVE_PATH)

    # Save history
    history_df = pd.DataFrame(history_records)
    history_df.to_csv(METRICS_SAVE_PATH, index=False)
    print(f"\nSaved training history to {METRICS_SAVE_PATH}", flush=True)

    # Final Test Set Evaluation
    print("\n" + "=" * 80, flush=True)
    print("FINAL TEST EVALUATION ON UNSEEN TEST ASPECTS (Loading Best Checkpoint):", flush=True)
    print("=" * 80, flush=True)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_metrics, test_records = evaluate_feature_regressor(model, test_data, device)
    train_metrics, _ = evaluate_feature_regressor(model, train_data, device)
    val_metrics, _ = evaluate_feature_regressor(model, val_data, device)

    all_metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(TEST_METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    test_pred_df = pd.DataFrame(test_records)
    test_pred_df.to_csv(PREDICTIONS_COMPARISON_PATH, index=False)
    print(f"Saved test predictions comparison to {PREDICTIONS_COMPARISON_PATH}", flush=True)

    # Print Formatted Evaluation Report
    print(f"\n--- TRAIN / VAL / TEST PERFORMANCE COMPARISON ---", flush=True)
    print(f"{'Metric':<25} | {'Train':<10} | {'Validation':<10} | {'Test':<10}", flush=True)
    print("-" * 65, flush=True)
    print(f"{'Valence RMSE':<25} | {train_metrics['valence_rmse']:<10.4f} | {val_metrics['valence_rmse']:<10.4f} | {test_metrics['valence_rmse']:<10.4f}", flush=True)
    print(f"{'Arousal RMSE':<25} | {train_metrics['arousal_rmse']:<10.4f} | {val_metrics['arousal_rmse']:<10.4f} | {test_metrics['arousal_rmse']:<10.4f}", flush=True)
    print(f"{'Overall RMSE':<25} | {train_metrics['overall_rmse']:<10.4f} | {val_metrics['overall_rmse']:<10.4f} | {test_metrics['overall_rmse']:<10.4f}", flush=True)
    print(f"{'Valence MAE':<25} | {train_metrics['valence_mae']:<10.4f} | {val_metrics['valence_mae']:<10.4f} | {test_metrics['valence_mae']:<10.4f}", flush=True)
    print(f"{'Arousal MAE':<25} | {train_metrics['arousal_mae']:<10.4f} | {val_metrics['arousal_mae']:<10.4f} | {test_metrics['arousal_mae']:<10.4f}", flush=True)
    print(f"{'Valence R²':<25} | {train_metrics['valence_r2']:<10.4f} | {val_metrics['valence_r2']:<10.4f} | {test_metrics['valence_r2']:<10.4f}", flush=True)
    print(f"{'Arousal R²':<25} | {train_metrics['arousal_r2']:<10.4f} | {val_metrics['arousal_r2']:<10.4f} | {test_metrics['arousal_r2']:<10.4f}", flush=True)
    print(f"{'Valence Lin CCC':<25} | {train_metrics['valence_ccc']:<10.4f} | {val_metrics['valence_ccc']:<10.4f} | {test_metrics['valence_ccc']:<10.4f}", flush=True)
    print(f"{'Arousal Lin CCC':<25} | {train_metrics['arousal_ccc']:<10.4f} | {val_metrics['arousal_ccc']:<10.4f} | {test_metrics['arousal_ccc']:<10.4f}", flush=True)
    print("=" * 65, flush=True)

    # Print Sample Ground Truth vs Predictions
    print("\nSAMPLE GROUND TRUTH VS PREDICTIONS (Test Set):", flush=True)
    print(f"{'Sentence (Snippet)':<30} | {'Aspect':<18} | {'True V':<7} | {'Pred V':<7} | {'Err V':<7} | {'True A':<7} | {'Pred A':<7}", flush=True)
    print("-" * 105, flush=True)
    for r in test_records[:12]:
        sent_clip = (r['sentence'][:27] + "...") if len(r['sentence']) > 27 else r['sentence']
        print(f"{sent_clip:<30} | {r['aspect']:<18} | {r['true_valence']:<7.3f} | {r['pred_valence']:<7.3f} | {r['valence_error']:<7.3f} | {r['true_arousal']:<7.3f} | {r['pred_arousal']:<7.3f}", flush=True)
    print("=" * 105, flush=True)

    return model, test_metrics, history_df


if __name__ == "__main__":
    train_aspect_emotion_regressor()
