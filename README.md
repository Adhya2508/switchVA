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
$$\begin{bmatrix} \hat{V} \\ \hat{A} \end{bmatrix} = \sigma\left( \mathbf{W}_{\text{head}} \mathbf{f}_{\text{fusion}} + (\mathbf{p}_{\text{opinion}} - 0.5) \odot \sigma(\mathbf{g}_{\text{prior}}) \cdot 6.0 \right)$$
where $\mathbf{g}_{\text{prior}}$ is initialised at $[0.85, 0.70]$ (v2 — strengthened from $[0.4, 0.3]$) to give the opinion lexicon prior strong influence.

### 6. Multi-Objective Loss Formulation (v2 — Contrastive-Aware)
$$\mathcal{L}_{\text{reg}} = \text{MSE}(V, \hat{V}) + \text{MSE}(A, \hat{A}) + 0.5 \cdot \text{SmoothL1}(V, \hat{V}) + 0.5 \cdot \text{SmoothL1}(A, \hat{A}) + 0.5 \cdot (1 - \text{CCC})$$
$$\mathcal{L}_{\text{contrastive}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j)\in\mathcal{P}} \max(0,\; 0.3 - |\hat{V}_i - \hat{V}_j|)$$
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reg}} + 0.4 \cdot \mathcal{L}_{\text{contrastive}}$$
where $\mathcal{P}$ is the set of same-sentence aspect pairs with opposite ground-truth polarities.

---

## 3. Quantitative Evaluation Benchmarks

The model was evaluated on unseen test sentences across standard continuous regression and concordance metrics:

| Evaluation Metric | Train Split | Validation Split | Test Split (Unseen Sentences) |
| :--- | :---: | :---: | :---: |
| **Overall RMSE** | **0.1605** | **0.1760** | **0.1697** |
| **Valence RMSE** | 0.1946 | 0.2113 | **0.2103** |
| **Arousal RMSE** | 0.1168 | 0.1315 | **0.1158** |
| **Valence MAE** | **0.1474** | 0.1765 | **0.1692** |
| **Arousal MAE** | **0.0936** | 0.1042 | **0.0889** |
| **Valence R²** | 0.3651 | 0.1589 | **0.2107** |
| **Arousal R²** | 0.2942 | −0.028 | **0.1280** |
| **Valence Lin's CCC** | **0.6002** | **0.4174** | **0.4751** |
| **Arousal Lin's CCC** | **0.5329** | **0.2599** | **0.4076** |

> v2 improvements vs v1: Arousal CCC **+0.08** (0.33→0.41), Arousal MAE **−0.019** (0.108→0.089), Overall RMSE **−0.008** (0.178→0.170)

---

## 4. Multi-Aspect Polarity Divergence Verification (v2 — All Correct)

All contradicting-sentiment sentences now produce correct, divergent polarity predictions:

| Sentence | Aspect | Opinion | Valence | Arousal | Polarity |
| :--- | :--- | :--- | :---: | :---: | :---: |
| food awesome tha service slow thi | **food** | awesome | 0.931 | 0.601 | ✅ Positive |
| food awesome tha service slow thi | **service** | slow | 0.366 | 0.420 | ✅ Negative |
| Service bahut badhiya hai lekin price kafi high hai | **service** | badhiya | 0.885 | 0.503 | ✅ Positive |
| Service bahut badhiya hai lekin price kafi high hai | **price** | kafi high | 0.382 | 0.607 | ✅ Negative |
| The battery life is amazing but the display is disappointing | **battery life** | amazing | 0.901 | 0.638 | ✅ Positive |
| The battery life is amazing but the display is disappointing | **display** | disappointing | 0.187 | 0.638 | ✅ Negative |
| acting mast thi but story bakwas lagi | **acting** | mast | 0.856 | 0.607 | ✅ Positive |
| acting mast thi but story bakwas lagi | **story** | bakwas | 0.135 | 0.784 | ✅ Negative |
| camera quality bahut achi hai battery bekar hai | **camera quality** | achi | 0.866 | 0.465 | ✅ Positive |
| camera quality bahut achi hai battery bekar hai | **battery** | bekar | 0.180 | 0.685 | ✅ Negative |
| match me batting zabardast thi bowling weak thi | **batting** | zabardast | 0.857 | 0.724 | ✅ Positive |
| match me batting zabardast thi bowling weak thi | **bowling** | weak | 0.373 | 0.446 | ✅ Negative |

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
│   └── training_history_aspect_level.csv  # 300-epoch training loss & convergence history (v2)
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
