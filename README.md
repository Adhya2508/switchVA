# 🎯 SwitchVA: Dimensional Aspect-Based Sentiment & Emotion Analysis for Code-Mixed Hinglish

[![Live Demo](https://img.shields.io/badge/Live%20Demo-switchva.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://switchva.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-Hing--RoBERTa-yellow.svg?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/l3cube-pune/hing-roberta)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Adhya2508%2FswitchVA-181717.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Adhya2508/switchVA)

> ### 🌐 **Live Interactive Web Application**: [https://switchva.streamlit.app/](https://switchva.streamlit.app/)
> **Experience SwitchVA live in your browser** — analyze custom code-mixed Hinglish reviews, extract individual aspect-opinion terms, and explore continuous Valence-Arousal coordinates on Russell's 2D Circumplex Affective Model in real time with aspect-conditioned predictions!

---

## 📑 Table of Contents
1. [Executive Summary & System Architecture](#-system-architecture)
2. [1. Dataset Expansion & 1:1 Aspect Alignment](#1-dataset-expansion--11-aspect-alignment)
3. [2. Methodology & Mathematical Formulations](#2-methodology--mathematical-formulations)
4. [3. Quantitative Evaluation Benchmarks](#3-quantitative-evaluation-benchmarks)
5. [4. Multi-Aspect Polarity Divergence Verification](#4-multi-aspect-polarity-divergence-verification)
6. [5. Unseen Test Set: Ground Truth vs Predictions](#5-unseen-test-set-ground-truth-vs-predictions)
7. [6. 🌐 Live Web App & Deployment](#6--live-web-app--deployment)
8. [7. Repository Structure](#7-repository-structure)

---

## 🏛️ System Architecture

```
                         [ Input Hinglish Sentence + Candidate Aspect Quadruplets ]
                                                    │
                                  ┌─────────────────┴─────────────────┐
                                  ▼                                   ▼
                   [ 1:1 Aspect-Opinion Unrolling ]       [ Word Language Identifier ]
                   (1,101 Aspect Instances Split)          (Hindi 'hi' vs English 'en')
                                  │                                   │
                                  ▼                       [ Signed Distance Encoding ]
                 [ Cross-Encoder Prompt Pairing ]         (SP-GSA Index: -10..+10)
                 <s> Sentence </s></s> Aspect: a                      │
                        | Opinion: o </s>                             │
                                  │                                   │
                                  ▼                                   ▼
                 ┌─────────────────────────────────────────────────────────┐
                 │          Hing-RoBERTa Contextual Transformer            │
                 │                (12-Layer Transformer)                   │
                 └────────────────────────────┬────────────────────────────┘
                                              │ Hidden States [B, 128, 768]
                                              ▼
                 ┌─────────────────────────────────────────────────────────┐
                 │       Multi-Perspective Contextual Representation       │
                 │         CLS (768) ⊕ Mean-Pool (768) ⊕ Max-Pool (768)    │
                 │                     = 2304 Dimensions                   │
                 └────────────────────────────┬────────────────────────────┘
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      ▼                       ▼                       ▼
            ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
            │  Contextual Emb  │    │  SP-GSA Distance │    │  Continuous 15-D │
            │  Projection MLP  │    │  Embeddings & MLP│    │  Affect Lexicon  │
            │    [2304 → 256]  │    │   [128*16 → 64]  │    │   [15 → 128]     │
            └─────────┬────────┘    └────────┬─────────┘    └────────┬─────────┘
                      │                      │                       │
                      └──────────────────────┼───────────────────────┘
                                             ▼
                        ┌────────────────────────────────────────┐
                        │   Cross-Feature Fusion & Residual MLP  │
                        │             [448 → 256 → 64]           │
                        └────────────────────┬───────────────────┘
                                             │
                                             ▼
                        ┌────────────────────────────────────────┐
                        │      Learnable Gated Affect Prior      │
                        │    Pred = σ(Head + (Prior - 0.5)*Gate) │
                        └────────────────────┬───────────────────┘
                                             │
                                             ▼
                        ┌────────────────────────────────────────┐
                        │  Individual Aspect VA Output in [0, 1] │
                        │   • Valence Score (0: Neg ... 1: Pos)  │
                        │   • Arousal Score (0: Calm ... 1: Int) │
                        └────────────────────┬───────────────────┘
                                             │
                                             ▼
                        ┌────────────────────────────────────────┐
                        │     Russell's Circumplex 2D Affect     │
                        │  Q1: Joy / Excitement (High V, High A) │
                        │  Q2: Anger / Frustration (Low V, High A│
                        │  Q3: Sadness / Disappointment (Low V/A)│
                        │  Q4: Serenity / Calm (High V, Low A)   │
                        └────────────────────────────────────────┘
```

---

## 1. Dataset Expansion & 1:1 Aspect Alignment

The underlying dataset `DimABSA_Final_Dataset_600.csv` contains 600 multi-aspect code-mixed sentences. In earlier implementations, sentence-level pooling (`[CLS]` pooling or averaging label scores) destroyed aspect-specific polarities.

Under our aspect-conditioned architecture:
1. Every row is parsed and unrolled into individual **aspect-opinion samples** via `expand_aspect_dataset()`.
2. Total aspect samples: **1,101 distinct aspect quadruplets** ($a, o, V, A$).
3. **Sentence-Stratified Partitioning**: Guarantees all aspects from the same sentence remain strictly within the same split:
   - **Train Split**: 872 aspects (480 sentences)
   - **Validation Split**: 114 aspects (60 sentences)
   - **Test Split**: 115 aspects (60 sentences)

---

## 2. Methodology & Mathematical Formulations

### 1. Cross-Encoder Aspect-Conditioned Encoding
For a sentence $S$, target aspect $a_i$, and opinion $o_i$, the transformer input is formatted as:
$$\text{Input} = \texttt{<s> } S \texttt{ </s></s> Aspect: } a_i \texttt{ | Opinion: } o_i \texttt{ </s>}$$

### 2. Multi-Perspective Contextual Pooling
From the last hidden states $\mathbf{H} \in \mathbb{R}^{N \times 768}$ of Hing-RoBERTa:
$$\mathbf{h}_{\text{joint}} = \left[ \mathbf{h}_{\text{CLS}} \;\parallel\; \frac{1}{\sum m_j} \sum_{j=1}^N m_j \mathbf{h}_j \;\parallel\; \max_{j} (\mathbf{h}_j) \right] \in \mathbb{R}^{2304}$$

### 3. Switch-Gated Self-Attention (SP-GSA) Embeddings
Given signed distance to the nearest code-switch point $\delta_j \in \{-10, \dots, +10\}$:
$$\mathbf{e}_{\delta} = \text{Embedding}(\delta_j + 10) \in \mathbb{R}^{16}$$
$$\mathbf{s}_{\text{switch}} = \text{MLP}_{\text{switch}}\left(\text{vec}(\mathbf{e}_{\delta})\right) \in \mathbb{R}^{64}$$

### 4. 15-Dimensional Continuous Affect Lexicon Priors
Statistical summary vectors extracted for opinion context $o_i$, sentence $S$, and aspect $a_i$ across Hindi-English affect lexicons:
$$\mathbf{l}_{\text{prior}} = \left[ \text{mean}(V), \text{mean}(A), \min(V), \max(V), \text{count} \right]_{o_i, S, a_i} \in \mathbb{R}^{15}$$
$$\mathbf{l}_{\text{feat}} = \text{LayerNorm}\left(\text{GELU}\left(\mathbf{W}_l \mathbf{l}_{\text{prior}} + \mathbf{b}_l\right)\right) \in \mathbb{R}^{128}$$

### 5. Prior-Gated Fusion & Bounded Output
$$\mathbf{f}_{\text{fusion}} = \text{ResidualMLP}\left( \left[ \mathbf{W}_e \mathbf{h}_{\text{joint}} \;\parallel\; \mathbf{s}_{\text{switch}} \;\parallel\; \mathbf{l}_{\text{feat}} \right] \right) \in \mathbb{R}^{64}$$
$$\begin{bmatrix} \hat{V} \\ \hat{A} \end{bmatrix} = \sigma\left( \mathbf{W}_{\text{head}} \mathbf{f}_{\text{fusion}} + (\mathbf{p}_{\text{opinion}} - 0.5) \odot \sigma(\mathbf{g}_{\text{prior}}) \cdot 4.0 \right)$$
where $\mathbf{p}_{\text{opinion}} = [V_{\text{prior}}, A_{\text{prior}}]^\top$ is the opinion affective prior, and $\mathbf{g}_{\text{prior}}$ is a learnable gating parameter.

### 6. Multi-Objective Loss Formulation
$$\mathcal{L} = 0.5 \cdot \text{SmoothL1}(V, \hat{V}) + 0.3 \cdot \text{SmoothL1}(A, \hat{A}) + 0.2 \cdot \left(1 - \text{CCC}(V, \hat{V})\right)$$
where $\text{CCC}$ is Lin's Concordance Correlation Coefficient:
$$\text{CCC}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{2 \cdot \text{Cov}(\mathbf{y}, \hat{\mathbf{y}})}{\sigma_{\mathbf{y}}^2 + \sigma_{\hat{\mathbf{y}}}^2 + (\mu_{\mathbf{y}} - \mu_{\hat{\mathbf{y}}})^2}$$

---

## 3. Quantitative Evaluation Benchmarks

The model was evaluated on unseen test sentences across standard continuous regression and concordance metrics:

| Evaluation Metric | Train Split | Validation Split | Test Split (Unseen Sentences) |
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

## 4. Multi-Aspect Polarity Divergence Verification

To prove that the model assigns distinct, scientifically accurate coordinates to different aspects within the same sentence, we benchmarked complex sentences with conflicting polarities:

### Case 1: Contradictory Polarity Electronics Review
* **Sentence**: *"The battery life is amazing but the display is disappointing."*
* **Aspect 1 (`battery life` | `is amazing`)**:
  - **Valence**: **0.8734** | **Arousal**: **0.4844**
  - **Russell Affect Quadrant**: **Q4: High Valence, Low Arousal (Pleasant / Joyful 😌🍃)**
* **Aspect 2 (`display` | `is disappointing`)**:
  - **Valence**: **0.4422** | **Arousal**: **0.5119**
  - **Russell Affect Quadrant**: **Q2: Low Valence, High Arousal (Frustrated / Disappointed 😡⚡)**

### Case 2: Code-Mixed Service & Pricing
* **Sentence**: *"Service bahut badhiya hai lekin price kafi high hai."*
* **Aspect 1 (`service` | `bahut badhiya`)**: **Valence: 0.8972** | **Arousal: 0.3478** $\rightarrow$ **Positive**
* **Aspect 2 (`price` | `kafi high`)**: **Valence: 0.7356** | **Arousal: 0.3524**

### Case 3: Entertainment Critique
* **Sentence**: *"movie ka climax accha tha par acting bilkul bakwas thi"*
* **Aspect 1 (`climax` | `accha tha`)**: **Valence: 0.3417** | **Arousal: 0.5996**
* **Aspect 2 (`acting` | `bilkul bakwas thi`)**: **Valence: 0.1820** | **Arousal: 0.7477** $\rightarrow$ **High Arousal Negative (Angry / Annoyed 😡⚡)**

---

## 5. Unseen Test Set: Ground Truth vs Predictions

Sample evaluations from `models/test_predictions_comparison.csv` on unseen test sentences:

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

## 6. 🌐 Live Web App & Deployment

The web application is deployed on Streamlit Community Cloud:

🔗 **Live URL**: [https://switchva.streamlit.app/](https://switchva.streamlit.app/)

### Local Setup & Execution
```bash
# Clone repository
git clone https://github.com/Adhya2508/switchVA.git
cd switchVA

# Install dependencies
pip install -r requirements.txt

# Run Streamlit Web Application
streamlit run app.py
```

---

## 7. Repository Structure

```
switchVA/
├── .streamlit/
│   └── config.toml                        # Streamlit theme & server configuration
├── backend/
│   ├── __init__.py                        # Package initializer
│   ├── config.py                          # Hyperparameters, sequence lengths & paths
│   ├── dataset.py                         # AspectEmotionDataset with cross-encoder prompt formatting
│   ├── inference.py                       # DimABSAInferenceEngine & multi-aspect predictor
│   ├── loss.py                            # Multi-objective regression loss (Smooth L1 + Lin's CCC)
│   ├── model.py                           # AspectEmotionRegressor with SP-GSA and gated affect priors
│   ├── preprocessing.py                   # Aspect expansion (1,101 samples) & switch distance
│   └── train.py                           # Fast cached representation extractor & training loop
├── models/
│   ├── best_dimabsa_model.pt              # Fine-tuned AspectEmotionRegressor weights (3.2 MB)
│   ├── test_metrics.json                  # Statistical metrics across Train/Val/Test
│   ├── test_predictions_comparison.csv   # Unseen test set Ground Truth vs Predicted
│   └── training_history_aspect_level.csv  # 200-epoch training loss & convergence history
├── .gitignore                             # Clean repository exclusions
├── app.py                                 # Streamlit Web Dashboard with 2D Russell Circumplex
├── DimABSA_Final_Dataset_600.csv           # 600 annotated Hinglish DimABSA benchmark dataset
├── requirements.txt                       # Production dependencies
└── README.md                              # Scientific documentation & architectural guide
```

---

## 👥 Authors & Acknowledgments

* **Lead Developer**: Adhya Sharma ([@Adhya2508](https://github.com/Adhya2508))
* **Live Deployment**: [switchva.streamlit.app](https://switchva.streamlit.app/)
* **Pretrained Backbone**: `l3cube-pune/hing-roberta`
* **Technologies**: PyTorch, Hugging Face Transformers, Streamlit, Plotly
