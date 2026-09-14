import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "DimABSA_Final_Dataset_600.csv")
PRETRAINED_MODEL_PATH = os.path.join(BASE_DIR, "pretrained_models", "hing_roberta")
MODEL_NAME = PRETRAINED_MODEL_PATH if os.path.exists(PRETRAINED_MODEL_PATH) else "l3cube-pune/hing-roberta"
SAVE_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_dimabsa_model.pt")
CHECKPOINT_SAVE_PATH = os.path.join(SAVE_DIR, "DimABSA_Checkpoint_25Epochs.pt")
METRICS_SAVE_PATH = os.path.join(SAVE_DIR, "training_history_25epochs.csv")

os.makedirs(SAVE_DIR, exist_ok=True)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Hyperparameters
MAX_LEN = 128
MAX_DISTANCE = 5
NUM_DISTANCE_EMBEDDINGS = (2 * MAX_DISTANCE) + 1  # 11 (-5 to +5)
HIDDEN_SIZE = 768
NUM_HEADS = 8
NUM_RELATIONS = 4  # 0: Sequential, 1: Self-loop, 2: Switch Edge, 3: Aspect->Opinion
RGAT_LAYERS = 2

# Training Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
EPOCHS = 25
POS_WEIGHT = 25.0

# Inference & Decoding Hyperparameters
SPAN_THRESHOLD = 0.55
TOP_K_SPANS = 10
VALENCE_THRESHOLD = 0.50
AROUSAL_THRESHOLD = 0.50
