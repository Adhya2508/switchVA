# 🎯 SwitchVA: Dimensional Aspect-Based Sentiment & Emotion Analysis for Code-Mixed Hinglish

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-Hing--RoBERTa-yellow.svg?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/l3cube-pune/hing-roberta)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-3F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Adhya2508%2FswitchVA-181717.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Adhya2508/switchVA)

---

## 📌 Executive Summary

**SwitchVA** (Dimensional Aspect-Based Sentiment Analysis for Code-Mixed Text) is a deep neural framework engineered to solve the complex linguistic and affective challenges of **Hinglish (Hindi-English code-switched text)**. 

Unlike conventional sentiment analysis models that only classify coarse binary polarity (positive/negative), SwitchVA performs **fine-grained dimensional affective computing**:
1. **Identifies Aspect & Opinion Spans** (e.g., `"camera quality"` $\rightarrow$ `"bohot zabardast"`) via **Biaffine Span Extractors**.
2. **Models Code-Switching Dynamics** using a **Switch-Gated Self-Attention (SP-GSA)** layer parameterized by signed boundary distance embeddings.
3. **Encodes Higher-Order Linguistic Dependencies** via a **4-Relational Graph Attention Network (RGAT)** and **Cross-Attention Fusion**.
4. **Predicts Continuous Valence & Arousal (VA)** coordinates $[0.0, 1.0]$ mapped onto **Russell's Circumplex Model of Affect** (4 quadrants of fine-grained emotion).

---

## 🏛️ End-to-End System Architecture

```
                                  [ Input Sentence (Hinglish) ]
                                                │
                                 ┌──────────────┴──────────────┐
                                 ▼                             ▼
                     [ Subword Tokenization ]       [ Word Language Identifier ]
                     (Hing-RoBERTa Vocabulary)     (Hindi 'hi' vs English 'en')
                                 │                             │
                                 │                 [ Signed Distance Encoding ]
                                 │                 (Distance to switch: -5..+5)
                                 │                             │
                                 ▼                             ▼
                      ┌──────────────────────────────────────────┐
                      │    Hing-RoBERTa Contextual Backbone      │
                      │         (12-Layer Transformer)           │
                      └────────────────────┬─────────────────────┘
                                           │ Token Embeddings [B, N, 768]
                                           ▼
                      ┌──────────────────────────────────────────┐
                      │   Switch-Gated Self-Attention (SP-GSA)   │
                      │  Modulates representations near switches │
                      └─────────────┬────────────────────────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
     ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────┐
     │  Biaffine Span   │  │  Biaffine Span   │  │  4-Relational Graph    │
     │  Aspect Extractor│  │ Opinion Extractor│  │     Construction       │
     │   [B, N, N]      │  │   [B, N, N]      │  │  - Sequential Edges    │
     └─────────┬────────┘  └────────┬─────────┘  │  - Self Loops          │
               │                    │            │  - Switch Edges        │
               │ (NMS Candidate Spans)           │  - Aspect->Opinion Rel │
               │                    │            └───────────┬────────────┘
               └──────────┬─────────┘                        │
                          │                                  ▼
                          │                     ┌────────────────────────┐
                          │                     │   Relational GAT       │
                          │                     │ (2-Layer RGAT Network) │
                          │                     └────────────┬───────────┘
                          │                                  │ Graph Embeddings
                          └─────────────────┬────────────────┘
                                            ▼
                               ┌───────────────────────────┐
                               │   Cross-Attention Fusion  │
                               │  (Multi-Head Interaction) │
                               └────────────┬──────────────┘
                                            │ Fused Representation [CLS]
                                            ▼
                               ┌───────────────────────────┐
                               │  Continuous Regression    │
                               │          Heads            │
                               │  ┌─────────────────────┐  │
                               │  │ Valence MLP (0 - 1) │  │
                               │  │ Arousal MLP (0 - 1) │  │
                               │  └─────────────────────┘  │
                               └────────────┬──────────────┘
                                            │
                                            ▼
                               ┌───────────────────────────┐
                               │ Russell's Circumplex Map  │
                               │  Q1: Joyful / Excited     │
                               │  Q2: Frustrated / Angry   │
                               │  Q3: Disappointed / Sad   │
                               │  Q4: Content / Relaxed    │
                               └───────────────────────────┘
```

---

## 🔬 Core Methodological Innovations

### 1. Contextual Backbone (`Hing-RoBERTa`)
Employs `l3cube-pune/hing-roberta`, a specialized RoBERTa transformer pre-trained on over 1.4 billion Hindi-English code-mixed tokens. It natively handles Romanized Hindi subwords (e.g., `bohot`, `achha`, `chalega`, `bakwas`).

### 2. Switch-Gated Self-Attention (SP-GSA)
Code-switching points often carry intense emotional sentiment shifts (e.g., *"Phone is nice lekin battery bohot bekar hai"*).
- For each token $i$, we compute its signed distance $\delta_i \in \{-5, \dots, +5\}$ to the nearest language switch point.
- Distance embedding $\mathbf{e}_{\delta_i} \in \mathbb{R}^H$ is projected alongside hidden state $\mathbf{h}_i$:
$$\mathbf{g}_i = \sigma\left(\mathbf{W}_g [\mathbf{h}_i \,\|\, \mathbf{e}_{\delta_i}] + \mathbf{b}_g\right)$$
$$\mathbf{h}_i' = \mathbf{h}_i + \mathbf{g}_i \odot \mathbf{h}_i$$

### 3. Biaffine Span Extractors with Non-Maximum Suppression (NMS)
Aspect terms and opinion expressions are extracted without restrictive sequential CRF limitations:
$$\mathbf{S}_{i, j} = (\mathbf{W}_{\text{start}} \mathbf{h}_i')^\top \mathbf{U} (\mathbf{W}_{\text{end}} \mathbf{h}_j')$$
Where $\mathbf{S} \in \mathbb{R}^{N \times N}$ is an upper-triangular span scoring matrix. Candidate spans above threshold $\tau = 0.55$ are decoded using overlap-based **Non-Maximum Suppression (NMS)**.

### 4. 4-Relational Graph Attention Network (RGAT)
Constructs a heterogeneous token-level graph with 4 distinct relation types $\mathcal{R}$:
* **Relation 0 (Sequential Edges)**: $i \leftrightarrow i+1$ linear sentence structure.
* **Relation 1 (Self Loops)**: $i \leftrightarrow i$ identity maintenance.
* **Relation 2 (Switch Edges)**: Connected token pairs crossing a language boundary.
* **Relation 3 (Aspect $\leftrightarrow$ Opinion Cross Edges)**: Semantic affective dependencies connecting identified aspect and opinion candidate spans.

The relational attention weight between tokens $i$ and $j$ under relation $r_{ij}$ is computed as:
$$\alpha_{ij} = \text{Softmax}_j \left(\text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{W} \mathbf{h}_i \,\|\, \mathbf{W} \mathbf{h}_j \,\|\, \mathbf{e}_{r_{ij}}]\right)\right)$$

### 5. Multi-Task Unified Loss Formulation
Trained end-to-end using a composite loss balancing classification sparsity and continuous metric alignment:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{span}} + 0.5 \cdot \mathcal{L}_{\text{regression}} + 0.5 \cdot \mathcal{L}_{\text{CCC}}^{\text{Valence}} + 0.5 \cdot \mathcal{L}_{\text{CCC}}^{\text{Arousal}}$$

- **Weighted BCE for Spans**: Handles sparse positive span matrices with $\text{pos\_weight} = 25.0$.
- **Smooth L1 (Huber) Regression**: Robust against outlier annotations.
- **Lin's Concordance Correlation Coefficient (CCC)**: Maximizes both scale invariance and ranking order correlation:
$$\text{CCC}(y, \hat{y}) = \frac{2 \rho \sigma_y \sigma_{\hat{y}}}{\sigma_y^2 + \sigma_{\hat{y}}^2 + (\mu_y - \mu_{\hat{y}})^2}, \quad \mathcal{L}_{\text{CCC}} = 1 - \text{CCC}$$

---

## 📊 Training Progression (25 Epochs)

The model converged smoothly across 25 training epochs on the curated DimABSA dataset:

| Metric | Epoch 1 (Initial) | Epoch 10 (Midway) | Epoch 25 (Final Best) |
| :--- | :---: | :---: | :---: |
| **Training Loss** | `2.4788` | `0.8707` | **`0.3648`** |
| **Validation Loss** | `2.3538` | `1.0297` | **`0.7058`** |
| **Loss Reduction** | Baseline | $-64.1\%$ | **$-78.9\%$** |

```
Train vs Validation Loss Curve:
Loss
2.5 | █ 
2.0 |  █ 
1.5 |   █ █ 
1.0 |      █ █ 
0.5 |         █ █ █ █ █ █ █ █ (Train: 0.3648 / Val: 0.7058)
0.0 └──────────────────────────── Epochs (1 -> 25)
```

---

## 💻 Streamlit Web Application Features

The interactive dashboard (`app.py`) provides:
* ⚡ **Live Hinglish Sentence Analyzer**: Real-time word language tagging (`[HI]` vs `[EN]`), switch distance display, aspect & opinion extraction pills, and continuous Valence-Arousal gauge meters.
* 🧭 **Russell's Circumplex Affect Visualizer**: 2D scatter quadrant mapping (Q1: Joy/Excitement, Q2: Frustration/Anger, Q3: Sadness/Disappointment, Q4: Contentment/Calm).
* 🗂️ **Batch Dataset Explorer**: Search, filter, and inspect 600 annotated Hinglish samples with distribution histograms.
* 📈 **Training Dynamics & Loss Inspection**: Interactive Plotly charts of 25-epoch train/val loss curves.
* 🕸️ **Model Architecture & Graph Visualizer**: Deep-dive component explanations and relational schema.

---

## 📁 Repository Structure

```
switchVA/
├── .streamlit/
│   └── config.toml                  # Streamlit theme & UI styling configuration
├── backend/
│   ├── __init__.py                  # Backend package initializer
│   ├── config.py                    # Hyperparameters, paths, and device settings
│   ├── dataset.py                   # PyTorch Dataset & DataLoader with padding & collate
│   ├── inference.py                 # DimABSAInferenceEngine & NMS decoding pipeline
│   ├── loss.py                      # Multi-task loss (Weighted BCE + Smooth L1 + CCC)
│   ├── model.py                     # SP-GSA, Biaffine Heads, RGAT Network, Fusion, MLP
│   ├── preprocessing.py             # Language tagger, switch distance calculator
│   └── train.py                     # 25-Epoch training and validation runner
├── models/
│   └── training_history_25epochs.csv# Stored training loss and validation loss metrics
├── .gitignore                       # Clean repository exclusions (large binary weights)
├── app.py                           # Full-featured Streamlit Web Dashboard
├── DimABSA_Final_Dataset_600.csv     # 600 annotated Hinglish DimABSA benchmark dataset
├── requirements.txt                 # Python dependencies for local and cloud deployment
└── README.md                        # Comprehensive system architecture & deployment guide
```

---

## 🚀 Local Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Adhya2508/switchVA.git
cd switchVA
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
py -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## ☁️ How to Deploy on Streamlit Community Cloud

Deploying SwitchVA to the web for free with **Streamlit Community Cloud** takes only 2 minutes:

### Step 1: Push Code to GitHub
Ensure all files are committed and pushed to `https://github.com/Adhya2508/switchVA` on the `main` branch:
```bash
git add .
git commit -m "Deploy SwitchVA application"
git branch -M main
git push -u origin main
```

### Step 2: Sign in to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Click **"Continue with GitHub"** and authorize Streamlit with your GitHub account (`Adhya2508`).

### Step 3: Deploy New App
1. Click the **"New app"** button in the top right.
2. In the deployment modal, enter the following configuration:
   * **Repository**: `Adhya2508/switchVA`
   * **Branch**: `main`
   * **Main file path**: `app.py`
   * **App URL (optional)**: `switchva-hinglish-dimabsa.streamlit.app`
3. Click **"Deploy!"**.

### Step 4: Live Build & Verification
* Streamlit Cloud will automatically read `requirements.txt`, install PyTorch, Transformers, Plotly, and Streamlit, and launch the application.
* When first analyzing a sentence, `transformers` will automatically download `l3cube-pune/hing-roberta` directly from Hugging Face Hub.

---

## 📜 Dataset Reference

The project uses `DimABSA_Final_Dataset_600.csv` containing:
* `Sentence`: Code-switched Hinglish text
* `Word_Language_Labels`: Token-level language tags (`[HI]`, `[EN]`, `[O]`)
* `Aspect_Spans`: Ground-truth target aspect phrases
* `Opinion_Spans`: Ground-truth opinion phrases
* `Valence_Score`: Ground-truth continuous valence rating $[0.0 - 1.0]$
* `Arousal_Score`: Ground-truth continuous arousal rating $[0.0 - 1.0]$

---

## 👥 Authors & Acknowledgments

* **Developer**: Adhya Sharma ([@Adhya2508](https://github.com/Adhya2508))
* **Base Foundation**: Pre-trained on `l3cube-pune/hing-roberta`
* **Frameworks**: PyTorch, Hugging Face Transformers, Streamlit, Plotly
