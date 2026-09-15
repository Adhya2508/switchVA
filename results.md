# Dimensional Aspect-Based Sentiment Analysis (DimABSA) — NSSG-DimNet Empirical Results

---

## 🏛️ 1. NSSG-DimNet System Architecture

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

## 📊 2. Quantitative Evaluation Benchmarks

Dataset: **1,101 aspect-specific samples** (600 code-mixed sentences), split into Train (872 aspects / 480 sents), Validation (114 aspects / 60 sents), and Unseen Test (115 aspects / 60 sents).

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

## 🎯 3. Multi-Aspect Contrastive Sentiment Verification

The model correctly disentangles contrasting polarities in complex multi-clause Hinglish sentences without sentiment leakage between aspects:

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

## 🔬 4. Unseen Test Set: Ground Truth vs Predictions

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

## 🚀 5. How to Run

```bash
# Clone the repository
git clone https://github.com/Adhya2508/switchVA.git
cd switchVA

# Install dependencies
pip install -r requirements.txt

# Run the interactive Streamlit application
streamlit run app.py
```

*NSSG-DimNet v3 (Neuro-Symbolic Switch-Gated Dual-Graph Network) — September 2026*
