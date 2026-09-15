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
from backend.dataset import NSSGAspectDataset, nssg_collate_fn
from backend.model import NSSGDimNet
from backend.loss import (
    compute_aspect_regression_loss,
    calculate_comprehensive_metrics,
)

FEATURE_CACHE_PATH = os.path.join(SAVE_DIR, "cached_nssg_features.pt")


def build_contrastive_pairs(sentences, targets):
    """
    Precomputes static indices of sample pairs from the same sentence that have
    opposing ground truth polarities. Vectorized for instant loss calculation.
    """
    pair_i, pair_j = [], []
    sent_map = {}
    for idx, sent in enumerate(sentences):
        if sent not in sent_map:
            sent_map[sent] = []
        sent_map[sent].append(idx)

    for indices in sent_map.values():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                ii, jj = indices[i], indices[j]
                if (targets[ii, 0] - 0.5) * (targets[jj, 0] - 0.5) < 0:
                    pair_i.append(ii)
                    pair_j.append(jj)

    if pair_i:
        return torch.tensor(pair_i, dtype=torch.long), torch.tensor(pair_j, dtype=torch.long)
    return None, None


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
        f"NSSG-DimNet Splits: Train={len(train_df)} aspects ({len(train_sids)} sents), "
        f"Val={len(val_df)} aspects ({len(val_sids)} sents), "
        f"Test={len(test_df)} aspects ({len(test_sids)} sents)",
        flush=True,
    )

    return train_df, val_df, test_df


def extract_nssg_tensors(loader, transformer, device):
    all_token_embs = []
    all_switch_ids = []
    all_adjs = []
    all_lex_tokens = []
    all_asp_masks = []
    all_op_masks = []
    all_attn_masks = []
    all_op_priors = []
    all_targets = []
    sample_ids = []
    sentences = []
    aspects = []
    opinions = []

    with torch.inference_mode():
        for b in loader:
            inp = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            out = transformer(input_ids=inp, attention_mask=mask)
            token_embs = out.last_hidden_state.cpu()

            all_token_embs.append(token_embs)
            all_switch_ids.append(b["switch_ids"].cpu())
            all_adjs.append(b["adj"].cpu())
            all_lex_tokens.append(b["lex_tokens"].cpu())
            all_asp_masks.append(b["asp_mask"].cpu())
            all_op_masks.append(b["op_mask"].cpu())
            all_attn_masks.append(b["attention_mask"].cpu())
            all_op_priors.append(b["op_prior"].cpu())
            all_targets.append(b["target"].cpu())
            sample_ids.extend(b["sample_id"])
            sentences.extend(b["sentence"])
            aspects.extend(b["aspect"])
            opinions.extend(b["opinion"])

    return {
        "token_embs": torch.cat(all_token_embs, dim=0),
        "switch_ids": torch.cat(all_switch_ids, dim=0),
        "adjs": torch.cat(all_adjs, dim=0),
        "lex_tokens": torch.cat(all_lex_tokens, dim=0),
        "asp_masks": torch.cat(all_asp_masks, dim=0),
        "op_masks": torch.cat(all_op_masks, dim=0),
        "attn_masks": torch.cat(all_attn_masks, dim=0),
        "op_priors": torch.cat(all_op_priors, dim=0),
        "targets": torch.cat(all_targets, dim=0),
        "sample_ids": sample_ids,
        "sentences": sentences,
        "aspects": aspects,
        "opinions": opinions,
    }


def evaluate_nssg_model(model, data, device):
    model.eval()
    token_embs = data["token_embs"].to(device)
    switch_ids = data["switch_ids"].to(device)
    adjs = data["adjs"].to(device)
    lex_tokens = data["lex_tokens"].to(device)
    asp_masks = data["asp_masks"].to(device)
    op_masks = data["op_masks"].to(device)
    attn_masks = data["attn_masks"].to(device)
    op_priors = data["op_priors"].to(device)
    targets = data["targets"].to(device)

    with torch.no_grad():
        outputs = model(
            token_embs=token_embs,
            switch_ids=switch_ids,
            adj_matrices=adjs,
            lex_token_feats=lex_tokens,
            asp_mask=asp_masks,
            op_mask=op_masks,
            attention_mask=attn_masks,
            opinion_lex_prior=op_priors,
        )
        pred_val = outputs["valence"]
        pred_aro = outputs["arousal"]

        loss_dict = compute_aspect_regression_loss(
            pred_val=pred_val,
            pred_aro=pred_aro,
            target_val=targets[:, 0],
            target_aro=targets[:, 1],
        )

        pv = pred_val.cpu().numpy()
        pa = pred_aro.cpu().numpy()
        tv = targets[:, 0].cpu().numpy()
        ta = targets[:, 1].cpu().numpy()

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


def train_nssg_dimnet(
    epochs: int = 50,
    lr: float = 2.0e-3,
    device: torch.device = DEVICE,
):
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_df, val_df, test_df = prepare_aspect_splits()

    model_source = PRETRAINED_MODEL_PATH if os.path.exists(PRETRAINED_MODEL_PATH) else "l3cube-pune/hing-roberta"
    tokenizer = AutoTokenizer.from_pretrained(model_source)

    if os.path.exists(FEATURE_CACHE_PATH):
        print(f"Loading cached NSSG graph tensors from {FEATURE_CACHE_PATH}...", flush=True)
        cached = torch.load(FEATURE_CACHE_PATH, map_location="cpu")
        train_data = cached["train"]
        val_data = cached["val"]
        test_data = cached["test"]
    else:
        print(f"Extracting contextual and heterogeneous graph representations...", flush=True)
        transformer = AutoModel.from_pretrained(model_source).to(device)
        transformer.eval()

        train_ds = NSSGAspectDataset(train_df, tokenizer=tokenizer, max_len=MAX_LEN)
        val_ds = NSSGAspectDataset(val_df, tokenizer=tokenizer, max_len=MAX_LEN)
        test_ds = NSSGAspectDataset(test_df, tokenizer=tokenizer, max_len=MAX_LEN)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn=nssg_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=nssg_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=nssg_collate_fn)

        train_data = extract_nssg_tensors(train_loader, transformer, device)
        val_data = extract_nssg_tensors(val_loader, transformer, device)
        test_data = extract_nssg_tensors(test_loader, transformer, device)

        torch.save({"train": train_data, "val": val_data, "test": test_data}, FEATURE_CACHE_PATH)
        print(f"Cached NSSG graph tensors saved to {FEATURE_CACHE_PATH}", flush=True)

    # Precompute static vectorized contrastive pair index tensors
    pair_i, pair_j = build_contrastive_pairs(train_data["sentences"], train_data["targets"])
    if pair_i is not None:
        pair_i = pair_i.to(device)
        pair_j = pair_j.to(device)
        print(f"Precomputed {len(pair_i)} contrastive sentence pairs for divergent polarity training.", flush=True)

    # Initialize NSSG-DimNet
    model = NSSGDimNet(
        hidden_dim=768,
        switch_dim=64,
        span_dim=128,
        graph_dim=128,
        lex_dim=3,
        max_dist=MAX_DISTANCE,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_rmse = float("inf")
    history_records = []

    token_embs_tr = train_data["token_embs"].to(device)
    switch_ids_tr = train_data["switch_ids"].to(device)
    adjs_tr = train_data["adjs"].to(device)
    lex_tokens_tr = train_data["lex_tokens"].to(device)
    asp_masks_tr = train_data["asp_masks"].to(device)
    op_masks_tr = train_data["op_masks"].to(device)
    attn_masks_tr = train_data["attn_masks"].to(device)
    op_priors_tr = train_data["op_priors"].to(device)
    targets_tr = train_data["targets"].to(device)

    print("\n" + "=" * 80, flush=True)
    print(f"STARTING NSSG-DimNet TRAINING ({epochs} Epochs on {device})", flush=True)
    print("=" * 80, flush=True)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(
            token_embs=token_embs_tr,
            switch_ids=switch_ids_tr,
            adj_matrices=adjs_tr,
            lex_token_feats=lex_tokens_tr,
            asp_mask=asp_masks_tr,
            op_mask=op_masks_tr,
            attention_mask=attn_masks_tr,
            opinion_lex_prior=op_priors_tr,
        )

        loss_dict = compute_aspect_regression_loss(
            pred_val=outputs["valence"],
            pred_aro=outputs["arousal"],
            target_val=targets_tr[:, 0],
            target_aro=targets_tr[:, 1],
        )

        # Fast vectorized contrastive margin loss
        if pair_i is not None and len(pair_i) > 0:
            gaps = torch.abs(outputs["valence"][pair_i] - outputs["valence"][pair_j])
            c_loss = torch.clamp(0.35 - gaps, min=0.0).mean()
        else:
            c_loss = torch.tensor(0.0, device=device)

        total_loss = loss_dict["loss"] + 0.5 * c_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        val_metrics, _ = evaluate_nssg_model(model, val_data, device)

        history_records.append({
            "Epoch": epoch + 1,
            "Train Loss": round(float(total_loss.item()), 4),
            "Val Loss": val_metrics["loss"],
            "Val Valence RMSE": val_metrics["valence_rmse"],
            "Val Arousal RMSE": val_metrics["arousal_rmse"],
            "Val Overall RMSE": val_metrics["overall_rmse"],
            "Val Valence MAE": val_metrics["valence_mae"],
            "Val Arousal MAE": val_metrics["arousal_mae"],
            "Val Valence R2": val_metrics["valence_r2"],
            "Val Arousal R2": val_metrics["arousal_r2"],
        })

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(
                f"Epoch [{epoch+1:03d}/{epochs:03d}] | "
                f"Train Loss: {total_loss.item():.4f} | "
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

    # Save training history
    history_df = pd.DataFrame(history_records)
    history_df.to_csv(METRICS_SAVE_PATH, index=False)
    print(f"\nSaved training history to {METRICS_SAVE_PATH}", flush=True)

    # Final Test Set Evaluation
    print("\n" + "=" * 80, flush=True)
    print("FINAL TEST EVALUATION ON UNSEEN TEST SET (Best NSSG-DimNet Checkpoint):", flush=True)
    print("=" * 80, flush=True)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_metrics, test_records = evaluate_nssg_model(model, test_data, device)
    train_metrics, _ = evaluate_nssg_model(model, train_data, device)
    val_metrics, _ = evaluate_nssg_model(model, val_data, device)

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
    print(f"\n--- NSSG-DimNet TRAIN / VAL / TEST PERFORMANCE COMPARISON ---", flush=True)
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
    print("\nSAMPLE GROUND TRUTH VS PREDICTIONS (Unseen Test Set):", flush=True)
    print(f"{'Sentence (Snippet)':<30} | {'Aspect':<18} | {'True V':<7} | {'Pred V':<7} | {'Err V':<7} | {'True A':<7} | {'Pred A':<7}", flush=True)
    print("-" * 105, flush=True)
    for r in test_records[:12]:
        sent_clip = (r['sentence'][:27] + "...") if len(r['sentence']) > 27 else r['sentence']
        print(f"{sent_clip:<30} | {r['aspect']:<18} | {r['true_valence']:<7.3f} | {r['pred_valence']:<7.3f} | {r['valence_error']:<7.3f} | {r['true_arousal']:<7.3f} | {r['pred_arousal']:<7.3f}", flush=True)
    print("=" * 105, flush=True)

    return model, test_metrics, history_df


if __name__ == "__main__":
    train_nssg_dimnet()
