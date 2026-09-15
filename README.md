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
1. [NSSG-DimNet System Architecture](#-nssg-dimnet-system-architecture)
2. [Dataset Expansion & 1:1 Aspect Alignment](#1-dataset-expansion--11-aspect-alignment)
3. [Methodology & Mathematical Formulations](#2-methodology--mathematical-formulations)
4. [Quantitative Evaluation Benchmarks](#3-quantitative-evaluation-benchmarks)
5. [Multi-Aspect Polarity Divergence Verification](#4-multi-aspect-polarity-divergence-verification)
6. [Unseen Test Set: Ground Truth vs Predictions](#5-unseen-test-set-ground-truth-vs-predictions)
7. [Live Web App & Deployment](#6--live-web-app--deployment)
8. [Repository Structure](#7-repository-structure)

---

## 🏛️ NSSG-DimNet System Architecture

**NSSG-DimNet: Neuro-Symbolic Switch-Gated Dual-Graph Network for Hinglish DimABSA**

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                              INPUT LAYER                                               │
│   Tokens: ["srk", "ki", "acting", "mast", "thi", "but", "story", "bakwas", "lagi"]                     │
│   Language Identifiers: [EN, HI, EN, HI, HI, EN, EN, HI, HI]                                           │
│   Language Switch Distance ───► [ Switch Embedding Block ]                                             │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
┌───────────────────────────────────────────────────▼────────────────────────────────────────────────────┐
│                                    ENCODING & SWITCH-GATING LAYER                                      │
│   Multilingual Transformer Encoder (HingRoBERTa) + Switch Positional Embedding                         │
│                                           │                                                            │
│                                           ▼                                                            │
│                       [ Switch-Point Gated Attention Unit (SP-GSA) ]                                   │
│                 (Modulate token representations at language transition points)                         │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
┌──────────────────────────────────────┐          ┌──────────────────────────────────────────────────────┐
│     DUAL-BRANCH PROCESSING: BR 1     │          │             DUAL-BRANCH PROCESSING: BR 2             │
│    [ Biaffine Span-Pair Extractor ]  │          │   [ Heterogeneous Neuro-Symbolic Graph (RGAT) ]      │
│  • Target Aspect Span: "acting"      │          │   Edge Types:                                        │
│  • Opinion Span: "mast thi"          │          │     1. Syntactic Dependency Edges                    │
│                                      │          │     2. Code-Switch Transition Bridges                │
│                                      │          │     3. Semantic Aspect-Opinion Links                 │
│                                      │          │   + NRC-VAD / Hinglish Affect Lexicon Priors         │
└──────────────────┬───────────────────┘          └──────────────────────────┬───────────────────────────┘
                   │                                                         │
                   └────────────────────────────────┬────────────────────────┘
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                              FUSION LAYER                                              │
│                     [ Aspect-Guided Mutual Cross-Attention & Span Fusion ]                             │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
┌──────────────────────────────────────┐                          ┌──────────────────────────────────────┐
│       Valence Regression Head        │                          │        Arousal Regression Head       │
│  Continuous Valence: [0.00 to 1.00]  │                          │  Continuous Arousal: [0.00 to 1.00]  │
│        (Negative to Positive)        │                          │         (Passive to Excited)         │
└──────────────────────────────────────┘                          └──────────────────────────────────────┘
```

---

## 1. Dataset Expansion & 1:1 Aspect Alignment

The underlying dataset `DimABSA_Final_Dataset_600.csv` contains 600 multi-aspect code-mixed sentences. In naive implementations, sentence-level pooling (`[CLS]` pooling or averaging label scores) destroyed aspect-specific polarities.

Under our aspect-conditioned architecture:
1. Every row is parsed and unrolled into individual **aspect-opinion samples** via `expand_aspect_dataset()`.
2. Total aspect samples: **1,101 distinct aspect quadruplets** ($a, o, V, A$).
3. Sentence-level splits (80% Train, 10% Validation, 10% Test) ensure no sentence leakage across splits:
   - **Train**: 872 aspect instances (480 sentences)
   - **Validation**: 114 aspect instances (60 sentences)
   - **Test (Unseen)**: 115 aspect instances (60 sentences)

---

## 2. Methodology & Mathematical Formulations

### 2.1 Switch-Point Gated Attention (SP-GSA)
Given token representation $h_i$ and signed distance to nearest code-switching point $d_i$:
$$e_{sw}(i) = \text{Embedding}(d_i)$$
$$g_i = \sigma(W_g [h_i; e_{sw}(i)] + b_g)$$
$$h_i^{gated} = \text{LayerNorm}(h_i + g_i \odot h_i)$$

### 2.2 Biaffine Span-Pair Extraction
$$h_{start} = \text{MLP}_{start}(H_{gated}), \quad h_{end} = \text{MLP}_{end}(H_{gated})$$
$$S_{asp} = h_{start} W_{asp} h_{end}^T, \quad S_{op} = h_{start} W_{op} h_{end}^T$$
$$H_{span} = \text{MLP}_{span}([h_{asp}; h_{op}; h_{asp} \odot h_{op}])$$

### 2.3 Heterogeneous Relational Graph Module (RGAT)
Combines 3 edge types:
1. **Syntactic Dependency Edges**: local context window and syntactic head links.
2. **Code-Switch Transition Bridges**: edges connecting words across language boundaries ($\text{lang}(i) \neq \text{lang}(j)$).
3. **Semantic Aspect-Opinion Links**: bipartite links between aspect and opinion tokens.

$$\alpha_{ij}^r = \text{Softmax}_j\left(\text{LeakyReLU}(a_r^T [W_r h_i; W_r h_j])\right) \cdot A_{ij}^r$$
$$H_{graph} = \text{LayerNorm}\left(\sum_{r=0}^2 \sum_j \alpha_{ij}^r W_r h_j + W_{lex} e_{lex} + H_{gated}\right)$$

### 2.4 Multi-Objective Regression & Contrastive Margin Loss
$$\mathcal{L}_{total} = \lambda_{MSE} \mathcal{L}_{MSE} + \lambda_{Huber} \mathcal{L}_{Huber} + \lambda_{CCC} \mathcal{L}_{CCC} + \lambda_{contrast} \mathcal{L}_{contrast}$$
where $\mathcal{L}_{contrast} = \frac{1}{|\mathcal{P}|} \sum_{(i, j) \in \mathcal{P}} \max(0, m - |V_i - V_j|)$ for aspects from the same sentence with opposing ground-truth polarities.

---

## 3. Quantitative Evaluation Benchmarks

| Evaluation Metric | Train Split | Validation Split | Test Split (Unseen) |
| :--- | :---: | :---: | :---: |
| **Overall RMSE** | **0.1474** | **0.1707** | **0.1589** |
| **Valence RMSE** | **0.1813** | **0.2123** | **0.1940** |
| **Arousal RMSE** | **0.1028** | **0.1151** | **0.1133** |
| **Valence MAE** | **0.1403** | **0.1693** | **0.1540** |
| **Arousal MAE** | **0.0823** | **0.0934** | **0.0918** |
| **Valence R²** | **0.4488** | **0.1510** | **0.3278** |
| **Arousal R²** | **0.4535** | **0.2126** | **0.1647** |
| **Valence Lin's CCC** | **0.7095** | **0.5070** | **0.6431** |
| **Arousal Lin's CCC** | **0.6363** | **0.4531** | **0.4888** |

---

## 4. Multi-Aspect Polarity Divergence Verification

| Review Sentence | Aspect Span | Opinion Span | Valence (V) | Arousal (A) | Polarity & Quadrant |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `movie ka climax accha tha par acting bilkul bakwas thi` | **climax** | accha tha | **0.617** | **0.437** | **Positive** (Q4: High Valence, Low Arousal) |
| `movie ka climax accha tha par acting bilkul bakwas thi` | **acting** | bilkul bakwas thi | **0.204** | **0.677** | **Negative** (Q2: Low Valence, High Arousal) |
| `match me batting zabardast thi bowling weak thi` | **batting** | zabardast thi | **0.779** | **0.596** | **Positive** (Q1: High Valence, High Arousal) |
| `match me batting zabardast thi bowling weak thi` | **bowling** | weak thi | **0.372** | **0.481** | **Negative** (Q3: Low Valence, Low Arousal) |
| `food awesome tha service slow thi` | **food** | awesome tha | **0.822** | **0.635** | **Positive** (Q1: High Valence, High Arousal) |
| `food awesome tha service slow thi` | **service** | slow thi | **0.441** | **0.440** | **Negative** (Q3: Low Valence, Low Arousal) |
| `Service bahut badhiya hai lekin price kafi high hai` | **service** | bahut badhiya hai | **0.891** | **0.464** | **Positive** (Q4: High Valence, Low Arousal) |
| `Service bahut badhiya hai lekin price kafi high hai` | **price** | kafi high hai | **0.377** | **0.545** | **Negative** (Q2: Low Valence, High Arousal) |
| `screen acchi hai but battery backup bekar hai` | **screen** | acchi hai | **0.738** | **0.386** | **Positive** (Q4: High Valence, Low Arousal) |
| `screen acchi hai but battery backup bekar hai` | **battery backup** | bekar hai | **0.267** | **0.604** | **Negative** (Q2: Low Valence, High Arousal) |

---

## 5. Unseen Test Set: Ground Truth vs Predictions

| Sentence (Snippet) | Aspect | True V | Pred V | Err V | True A | Pred A | Err A |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `in se poochain ke jb last exam...` | exam | 0.220 | 0.321 | 0.101 | 0.780 | 0.621 | 0.159 |
| `aapne apne kafil ko kyu dhoka...` | kafil | 0.120 | 0.063 | 0.057 | 0.850 | 0.827 | 0.023 |
| `aapne apne kafil ko kyu dhoka...` | plan | 0.445 | 0.244 | 0.201 | 0.709 | 0.673 | 0.036 |
| `aapne apne kafil ko kyu dhoka...` | haram ka paisa | 0.100 | 0.062 | 0.038 | 0.880 | 0.838 | 0.042 |
| `aapne apne kafil ko kyu dhoka...` | halal ki roti | 0.140 | 0.207 | 0.067 | 0.820 | 0.688 | 0.132 |
| `ye bi sahi hai wohi baat hui...` | apne log | 0.650 | 0.613 | 0.037 | 0.400 | 0.498 | 0.098 |
| `main apne desh ke business ko...` | desh ke business | 0.761 | 0.887 | 0.126 | 0.548 | 0.515 | 0.033 |
| `main apne desh ke business ko...` | traders | 0.750 | 0.715 | 0.035 | 0.700 | 0.530 | 0.170 |
| `traders ko justice tbi milegi...` | justice | 0.421 | 0.744 | 0.323 | 0.544 | 0.491 | 0.053 |
| `traders ko justice tbi milegi...` | govt | 0.300 | 0.491 | 0.191 | 0.600 | 0.559 | 0.041 |
| `traders ko justice tbi milegi...` | business | 0.450 | 0.757 | 0.307 | 0.550 | 0.481 | 0.069 |
| `apna maal apne pas rakho...` | maal | 0.370 | 0.192 | 0.178 | 0.497 | 0.662 | 0.165 |

---

## 6. 🌐 Live Web App & Deployment

The system is deployed as a live cloud application:
🔗 **[https://switchva.streamlit.app/](https://switchva.streamlit.app/)**

### Features:
- **Sentence Analysis**: Live inference on arbitrary Hinglish sentences with automatic aspect-opinion span extraction and affect quadrant classification.
- **Benchmark Suite**: Quantitative comparison metrics across Train, Validation, and Test splits.
- **Model Evaluation**: Interactive 2D Russell's Circumplex scatter plot visualizing ground truth vs predicted coordinates.
- **Dataset Browser**: Searchable interface for the expanded 1,101-aspect dataset with linguistic statistics.

---

## 7. Repository Structure

```
f:\review 3\
├── backend/
│   ├── __init__.py
│   ├── config.py                 # Paths, hyper-parameters, and model dimensions
│   ├── preprocessing.py          # Language identification & switch distance encoding
│   ├── dataset.py                # NSSGAspectDataset & graph tensor constructors
│   ├── model.py                  # NSSGDimNet architecture (SP-GSA, Biaffine, RGAT, Fusion)
│   ├── loss.py                   # Multi-objective Huber/MSE/CCC loss & evaluation metrics
│   ├── train.py                  # Training pipeline with contrastive margin loss
│   └── inference.py              # NSSG-DimNet inference engine & clause analyzer
├── models/
│   ├── best_dimabsa_model.pt     # Fine-tuned NSSG-DimNet checkpoint weights
│   ├── dimabsa_checkpoint.pt     # Training optimizer and state checkpoint
│   ├── test_metrics.json         # Complete quantitative test metrics
│   ├── test_predictions_comparison.csv # Unseen test set predictions
│   └── training_history_aspect_level.csv # Epoch-by-epoch loss & RMSE trajectory
├── app.py                        # Streamlit web application frontend
├── results.md                    # Research and empirical evaluation report
├── requirements.txt              # Environment dependencies
└── README.md                     # Documentation
```

---

## 8. Quickstart

```bash
# Clone the repository
git clone https://github.com/Adhya2508/switchVA.git
cd switchVA

# Install dependencies
pip install -r requirements.txt

# Run the training script
python -m backend.train

# Launch the Streamlit application
streamlit run app.py
```
