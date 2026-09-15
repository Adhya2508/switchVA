import os
import torch

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "DimABSA_Final_Dataset_600.csv")
PRETRAINED_MODEL_PATH = os.path.join(BASE_DIR, "pretrained_models", "hing_roberta")
MODEL_NAME = PRETRAINED_MODEL_PATH if os.path.exists(PRETRAINED_MODEL_PATH) else "l3cube-pune/hing-roberta"
SAVE_DIR = os.path.join(BASE_DIR, "models")

# Model Save Paths
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_dimabsa_model.pt")
SPAN_MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_span_model.pt")
CHECKPOINT_SAVE_PATH = os.path.join(SAVE_DIR, "DimABSA_Checkpoint.pt")
METRICS_SAVE_PATH = os.path.join(SAVE_DIR, "training_history_aspect_level.csv")
TEST_METRICS_PATH = os.path.join(SAVE_DIR, "test_metrics.json")
PREDICTIONS_COMPARISON_PATH = os.path.join(SAVE_DIR, "test_predictions_comparison.csv")

os.makedirs(SAVE_DIR, exist_ok=True)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture Hyperparameters
MAX_LEN = 128
MAX_DISTANCE = 5
NUM_DISTANCE_EMBEDDINGS = (2 * MAX_DISTANCE) + 1  # 11 (-5 to +5)
HIDDEN_SIZE = 768
NUM_HEADS = 8
NUM_RELATIONS = 4
RGAT_LAYERS = 2
DROPOUT = 0.2

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
REG_HEAD_LR = 2e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 15
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0

# Loss Weights
LOSS_MSE_WEIGHT = 1.0
LOSS_HUBER_WEIGHT = 0.5
LOSS_CCC_WEIGHT = 0.5
POS_WEIGHT = 25.0

# Inference & Decoding Hyperparameters
SPAN_THRESHOLD = 0.50
TOP_K_SPANS = 10
VALENCE_THRESHOLD = 0.50
AROUSAL_THRESHOLD = 0.50
