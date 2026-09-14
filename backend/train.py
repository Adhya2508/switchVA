import os
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoTokenizer, AutoModel

from backend.config import (
    DATASET_PATH,
    MODEL_NAME,
    PRETRAINED_MODEL_PATH,
    SAVE_DIR,
    MODEL_SAVE_PATH,
    CHECKPOINT_SAVE_PATH,
    METRICS_SAVE_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    POS_WEIGHT,
    DEVICE,
    MAX_LEN,
    MAX_DISTANCE,
)
from backend.preprocessing import parse_set_string, parse_float_string
from backend.dataset import DimABSADataset, collate_fn
from backend.model import DimABSAModel
from backend.loss import compute_total_loss, ccc_loss


def prepare_data_splits(csv_path: str = DATASET_PATH, random_state: int = 42):
    df = pd.read_csv(csv_path)

    # Clean and parse columns
    df["aspect_list"] = df["all_aspects"].apply(parse_set_string)
    df["opinion_list"] = df["all_opinions"].apply(parse_set_string)
    df["valence_list"] = df["valence_scores"].apply(parse_float_string)
    df["arousal_list"] = df["arousal_scores"].apply(parse_float_string)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state)

    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


def span_accuracy(pred_logits: torch.Tensor, ground_truth: torch.Tensor):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    correct = (pred == ground_truth).float().mean()
    return correct.item()


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device = DEVICE):
    model.eval()
    total_loss = 0.0
    total_span_loss = 0.0
    total_reg_loss = 0.0
    aspect_accs = []
    opinion_accs = []

    pred_vals = []
    true_vals = []
    pred_aros = []
    true_aros = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "switch_ids": batch["switch_ids"].to(device),
                "aspect_matrix": batch["aspect_matrix"].to(device),
                "opinion_matrix": batch["opinion_matrix"].to(device),
                "valence": batch["valence"],
                "arousal": batch["arousal"],
                "words": batch["words"],
            }

            outputs = model(batch_dev)
            loss_dict = compute_total_loss(outputs, batch_dev, pos_weight=POS_WEIGHT)

            total_loss += loss_dict["loss"].item()
            total_span_loss += loss_dict["span_loss"].item()
            total_reg_loss += loss_dict["regression_loss"].item()

            aspect_accs.append(span_accuracy(outputs["aspect_scores"], batch_dev["aspect_matrix"]))
            opinion_accs.append(span_accuracy(outputs["opinion_scores"], batch_dev["opinion_matrix"]))

            pred_vals.extend(outputs["valence"].cpu().numpy().tolist())
            pred_aros.extend(outputs["arousal"].cpu().numpy().tolist())

            for v, a in zip(batch["valence"], batch["arousal"]):
                if len(v) == 0:
                    true_vals.append(0.5)
                else:
                    true_vals.append(float(v.mean()))

                if len(a) == 0:
                    true_aros.append(0.5)
                else:
                    true_aros.append(float(a.mean()))

    n = len(loader)
    avg_loss = total_loss / n if n > 0 else 0.0

    val_mae = mean_absolute_error(true_vals, pred_vals) if pred_vals else 0.0
    aro_mae = mean_absolute_error(true_aros, pred_aros) if pred_aros else 0.0
    val_rmse = np.sqrt(mean_squared_error(true_vals, pred_vals)) if pred_vals else 0.0
    aro_rmse = np.sqrt(mean_squared_error(true_aros, pred_aros)) if pred_aros else 0.0

    return {
        "loss": avg_loss,
        "span_loss": total_span_loss / n if n > 0 else 0.0,
        "regression_loss": total_reg_loss / n if n > 0 else 0.0,
        "aspect_acc": float(np.mean(aspect_accs)) if aspect_accs else 0.0,
        "opinion_acc": float(np.mean(opinion_accs)) if opinion_accs else 0.0,
        "valence_mae": float(val_mae),
        "arousal_mae": float(aro_mae),
        "valence_rmse": float(val_rmse),
        "arousal_rmse": float(aro_rmse),
    }


def train_dimabsa(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    device: torch.device = DEVICE,
    progress_callback=None,
):
    """
    Trains the complete DimABSA model pipeline.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_df, val_df, test_df = prepare_data_splits()

    model_source = PRETRAINED_MODEL_PATH if os.path.exists(PRETRAINED_MODEL_PATH) else "l3cube-pune/hing-roberta"
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    transformer = AutoModel.from_pretrained(model_source)

    train_dataset = DimABSADataset(train_df, tokenizer=tokenizer)
    val_dataset = DimABSADataset(val_df, tokenizer=tokenizer)
    test_dataset = DimABSADataset(test_df, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = DimABSAModel(
        transformer_model=transformer,
        hidden_size=transformer.config.hidden_size,
        max_distance=MAX_DISTANCE,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print(f"Starting training for {epochs} epochs on device: {device}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch_dev = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "switch_ids": batch["switch_ids"].to(device),
                "aspect_matrix": batch["aspect_matrix"].to(device),
                "opinion_matrix": batch["opinion_matrix"].to(device),
                "valence": batch["valence"],
                "arousal": batch["arousal"],
                "words": batch["words"],
            }

            optimizer.zero_grad()
            outputs = model(batch_dev)
            loss_dict = compute_total_loss(outputs, batch_dev, pos_weight=POS_WEIGHT)
            loss = loss_dict["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device=device)
        val_loss = val_metrics["loss"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Aspect Acc: {val_metrics['aspect_acc']:.4f} | Opinion Acc: {val_metrics['opinion_acc']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, CHECKPOINT_SAVE_PATH)

        if progress_callback:
            progress_callback(epoch + 1, epochs, train_loss, val_loss, val_metrics)

    # Save training history
    history_df = pd.DataFrame({
        "Epoch": list(range(1, epochs + 1)),
        "Train Loss": train_losses,
        "Validation Loss": val_losses,
    })
    history_df.to_csv(METRICS_SAVE_PATH, index=False)

    # Final Test Evaluation
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_metrics = evaluate(model, test_loader, device=device)
    print("\n" + "=" * 50)
    print("FINAL TEST EVALUATION METRICS:")
    print(f"Aspect Span Accuracy  : {test_metrics['aspect_acc']:.4f}")
    print(f"Opinion Span Accuracy : {test_metrics['opinion_acc']:.4f}")
    print(f"Valence MAE           : {test_metrics['valence_mae']:.4f}")
    print(f"Arousal MAE           : {test_metrics['arousal_mae']:.4f}")
    print(f"Valence RMSE          : {test_metrics['valence_rmse']:.4f}")
    print(f"Arousal RMSE          : {test_metrics['arousal_rmse']:.4f}")
    print("=" * 50)

    return model, test_metrics, history_df


if __name__ == "__main__":
    train_dimabsa()
