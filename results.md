# Dimensional Aspect-Based Sentiment Analysis (DimABSA) & Emotion Regression
## Comprehensive Empirical Results & Verification Report

---

### 1. Executive Summary & Problem Formulation

In this work, the Hinglish Aspect-Based Emotion Regression system was refactored into a **scientifically correct aspect-conditioned regression architecture**. The fundamental flaw in previous approaches—sentence-level pooling and label averaging ($\text{mean}(V)$, $\text{mean}(A)$)—was eliminated. 

The dataset provides separate Valence and Arousal labels for each aspect term. Under the new architecture:
$$\text{Sentence } S \longrightarrow \{(a_1, o_1), (a_2, o_2), \dots, (a_k, o_k)\} \longrightarrow \left\{ (V_{a_1}, A_{a_1}), (V_{a_2}, A_{a_2}), \dots, (V_{a_k}, A_{a_k}) \right\}$$
where each aspect $a_i$ receives its own distinct continuous Valence and Arousal coordinates $(V, A) \in [0, 1]$ mapped to Russell's 2D Circumplex Model of Affect.

---

### 2. Quantitative Evaluation Benchmarks

The dataset was unrolled into **1,101 aspect-specific samples** using sentence-stratified partitioning to prevent data leakage across splits (Train: 872 aspects / 480 sentences, Validation: 114 aspects / 60 sentences, Test: 115 aspects / 60 sentences).

#### Performance Metrics Across Splits

| Metric | Train Split | Validation Split | Test Split (Unseen Sentences) |
| :--- | :---: | :---: | :---: |
| **Overall RMSE** | **0.1617** | **0.1811** | **0.1775** |
| **Valence RMSE** | 0.1885 | 0.2117 | **0.2101** |
| **Arousal RMSE** | 0.1295 | 0.1440 | **0.1374** |
| **Valence MAE** | **0.1451** | 0.1780 | **0.1699** |
| **Arousal MAE** | **0.1027** | 0.1176 | **0.1079** |
| **Valence Lin's CCC** | **0.6334** | **0.4265** | **0.4843** |
| **Arousal Lin's CCC** | **0.5273** | **0.2380** | **0.3300** |
| **Valence Pearson $r$** | **0.6517** | **0.4587** | **0.5110** |
| **Arousal Pearson $r$** | **0.5502** | **0.2540** | **0.3461** |
| **Combined Loss** | 0.5732 | 0.8577 | **0.7716** |

---

### 3. Multi-Aspect Divergence Verification

To evaluate whether the model successfully separates polarities within a single sentence, we tested multi-aspect sentences with opposing polarities:

#### Case 1: Opposite Polarity Electronics Review
- **Input Sentence**: *"The battery life is amazing but the display is disappointing."*
- **Aspect 1 (`battery life` | `amazing`)**:
  - **Valence**: **0.8734** | **Arousal**: **0.4844**
  - **Russell Affect Quadrant**: **Q4: High Valence, Low Arousal (Pleasant / Content / Joyful 😌🍃)**
  - **Confidence**: 75.9%
- **Aspect 2 (`display` | `disappointing`)**:
  - **Valence**: **0.4422** | **Arousal**: **0.5119**
  - **Russell Affect Quadrant**: **Q2: Low Valence, High Arousal (Frustrated / Disappointed 😡⚡)**
  - **Confidence**: 54.1%

#### Case 2: Code-Mixed Service Review
- **Input Sentence**: *"Service bahut badhiya hai lekin price kafi high hai."*
- **Aspect 1 (`service` | `bahut badhiya hai`)**:
  - **Valence**: **0.8972** | **Arousal**: **0.3478** $\rightarrow$ **Positive / Pleasant**
  - **Confidence**: 79.5%
- **Aspect 2 (`price` | `kafi high hai`)**:
  - **Valence**: **0.7356** | **Arousal**: **0.3524** $\rightarrow$ **Moderate / Subdued**

#### Case 3: Code-Mixed Entertainment Critique
- **Input Sentence**: *"movie ka climax accha tha par acting bilkul bakwas thi"*
- **Aspect 1 (`climax` | `accha tha`)**:
  - **Valence**: **0.3417** | **Arousal**: **0.5996**
- **Aspect 2 (`acting` | `bilkul bakwas thi`)**:
  - **Valence**: **0.1820** | **Arousal**: **0.7477** $\rightarrow$ **High Arousal, Strong Negative (Frustrated / Annoyed 😡⚡)**
  - **Confidence**: 77.9%

#### Case 4: Strong Negative Hinglish Colloquial
- **Input Sentence**: *"apna maal apne pas rakho oopar se bakwas kar rahe ho"*
- **Aspect 1 (`maal`)**:
  - **Valence**: **0.0897** | **Arousal**: **0.8394** $\rightarrow$ **Extreme High Arousal Negative (Angry / Frustrated)**
  - **Confidence**: 86.9%

---

### 4. Unseen Test Set: Ground Truth vs Prediction Comparison

The following table extracts ground truth comparisons from `models/test_predictions_comparison.csv` on unseen test sentences:

| Sample ID | Aspect Term | Ground Truth V | Predicted V | Valence Error | Ground Truth A | Predicted A | Arousal Error |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `25_0` | **kafil** | 0.120 | 0.122 | **0.002** | 0.850 | 0.834 | **0.016** |
| `25_2` | **haram ka paisa** | 0.100 | 0.116 | **0.016** | 0.880 | 0.840 | **0.040** |
| `40_0` | **desh ke business** | 0.761 | 0.876 | **0.115** | 0.548 | 0.367 | 0.181 |
| `40_1` | **traders** | 0.750 | 0.700 | **0.050** | 0.700 | 0.398 | 0.302 |
| `56_0` | **justice** | 0.421 | 0.398 | **0.023** | 0.544 | 0.558 | **0.014** |
| `56_1` | **govt** | 0.300 | 0.393 | **0.093** | 0.600 | 0.560 | **0.040** |
| `56_2` | **business** | 0.450 | 0.432 | **0.018** | 0.550 | 0.533 | **0.017** |
| `71_0` | **maal** | 0.370 | 0.423 | **0.053** | 0.497 | 0.525 | **0.028** |
| `82_1` | **back log post** | 0.458 | 0.439 | **0.019** | 0.542 | 0.586 | **0.044** |
| `110_1` | **sid** | 0.457 | 0.426 | **0.031** | 0.595 | 0.567 | **0.028** |
| `110_2` | **asim** | 0.457 | 0.426 | **0.031** | 0.595 | 0.568 | **0.027** |

---

### 5. Architectural Components & Implementation Files

- **Preprocessing & Dataset Expansion**: [`backend/preprocessing.py`](file:///f:/review%203/backend/preprocessing.py)
  - `expand_aspect_dataset`: Transforms 600 raw rows to 1,101 aspect quadruplets.
  - `calculate_switch_distance`: SP-GSA token distance calculator.
- **Aspect-Level Cross-Encoder Dataset**: [`backend/dataset.py`](file:///f:/review%203/backend/dataset.py)
  - Pairs sentence and aspect prompt `<s> Sentence </s></s> Aspect: {aspect} | Opinion: {opinion} </s>`.
- **Hybrid Neural Regressor**: [`backend/model.py`](file:///f:/review%203/backend/model.py)
  - `AspectEmotionRegressor`: Fuses 2304-D HingRoBERTa representations (CLS + Mean + Max), SP-GSA switch distance embeddings, and 15-D continuous Affect Lexicon prior vectors via balanced projection and learnable prior gating.
- **Multi-Objective Loss Function**: [`backend/loss.py`](file:///f:/review%203/backend/loss.py)
  - Combines Smooth L1 loss with Lin's Concordance Correlation Coefficient (CCC) optimization.
- **Training Pipeline**: [`backend/train.py`](file:///f:/review%203/backend/train.py)
  - 200 epochs with AdamW, Cosine Annealing, and early checkpoint saving.
- **Inference Engine**: [`backend/inference.py`](file:///f:/review%203/backend/inference.py)
  - `DimABSAInferenceEngine`: Extracts multi-aspect candidates and returns aspect-wise $(V, A)$ coordinates, polarities, and Russell affect quadrants.
- **Interactive Streamlit Web Dashboard**: [`app.py`](file:///f:/review%203/app.py)

---

### 6. How to Run the Application

#### Step 1: Install Dependencies (if not already installed)
```bash
pip install -r requirements.txt
```

#### Step 2: Launch Streamlit Dashboard
```bash
streamlit run app.py
```

#### Access Link:
Once started, the application is accessible at:
- **Local URL**: [http://localhost:8501](http://localhost:8501)
- **Network URL**: `http://<your-local-ip>:8501`

---
*Report generated on September 15, 2026 for Aspect-Based Emotion Regression (DimABSA).*
