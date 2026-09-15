# Dimensional Aspect-Based Sentiment Analysis (DimABSA) — Empirical Results

---

### 1. Problem Formulation

The system performs **aspect-conditioned continuous Valence & Arousal regression** on code-mixed Hinglish.
Each aspect receives its own $(V, A) \in [0,1]^2$ coordinates mapped to **Russell's 2D Circumplex Model of Affect**.

---

### 2. Architecture Fixes (v2 — Contrastive-Aware)

**Phase 1 — Clause-Scoped Inference**
- Clause splitter now recognises Hinglish tense markers (`tha`, `thi`, `hai`, `hain`) as clause boundaries
- Each aspect is encoded using its **own clause** as `text_a` so the CLS embedding is locally scoped
- Opinion = highest-affect lexicon word in the aspect's clause

**Phase 2 — Contrastive Training Signal**
- `lex_gate` raised `[0.4, 0.3]` → `[0.85, 0.70]`, prior multiplier `4.0` → `6.0`
- **Contrastive pair penalty**: for same-sentence aspects with opposite true polarities,
  a margin loss pushes predicted valences apart by >= 0.30
- 300 epochs (up from 200)

---

### 3. Quantitative Metrics

Dataset: **1,101 aspect-specific samples**, sentence-stratified (Train 872 / Val 114 / Test 115).

| Metric | Train | Validation | Test (Unseen) |
| :--- | :---: | :---: | :---: |
| **Overall RMSE** | 0.1605 | 0.1760 | **0.1697** |
| **Valence RMSE** | 0.1946 | 0.2113 | **0.2103** |
| **Arousal RMSE** | 0.1168 | 0.1315 | **0.1158** |
| **Valence MAE** | 0.1474 | 0.1765 | **0.1692** |
| **Arousal MAE** | 0.0936 | 0.1042 | **0.0889** |
| **Valence R2** | 0.3651 | 0.1589 | **0.2107** |
| **Arousal R2** | 0.2942 | -0.0278 | **0.1280** |
| **Valence Lin CCC** | 0.6002 | 0.4174 | **0.4751** |
| **Arousal Lin CCC** | 0.5329 | 0.2599 | **0.4076** |

vs previous: Arousal CCC +0.08, Overall RMSE -0.008, Arousal MAE -0.019.

---

### 4. Contrastive Sentence Verification (All Correct)

| Sentence | Aspect | Opinion | V | A | Polarity |
| :--- | :--- | :--- | :---: | :---: | :---: |
| food awesome tha service slow thi | **food** | awesome | 0.931 | 0.601 | Positive |
| food awesome tha service slow thi | **service** | slow | 0.366 | 0.420 | Negative |
| Service bahut badhiya hai lekin price kafi high hai | **service** | badhiya | 0.885 | 0.503 | Positive |
| Service bahut badhiya hai lekin price kafi high hai | **price** | kafi high | 0.382 | 0.607 | Negative |
| The battery life is amazing but the display is disappointing | **battery life** | amazing | 0.901 | 0.638 | Positive |
| The battery life is amazing but the display is disappointing | **display** | disappointing | 0.187 | 0.638 | Negative |
| acting mast thi but story bakwas lagi | **acting** | mast | 0.856 | 0.607 | Positive |
| acting mast thi but story bakwas lagi | **story** | bakwas | 0.135 | 0.784 | Negative |
| camera quality bahut achi hai battery bekar hai | **camera quality** | achi | 0.866 | 0.465 | Positive |
| camera quality bahut achi hai battery bekar hai | **battery** | bekar | 0.180 | 0.685 | Negative |
| match me batting zabardast thi bowling weak thi | **batting** | zabardast | 0.857 | 0.724 | Positive |
| match me batting zabardast thi bowling weak thi | **bowling** | weak | 0.373 | 0.446 | Negative |

---

### 5. Test Set Ground Truth vs Predictions (Sample)

| Aspect | True V | Pred V | Err V | True A | Pred A | Err A |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| kafil | 0.120 | 0.070 | 0.050 | 0.850 | 0.876 | 0.026 |
| haram ka paisa | 0.100 | 0.053 | 0.047 | 0.880 | 0.897 | 0.017 |
| desh ke business | 0.761 | 0.831 | 0.070 | 0.548 | 0.531 | 0.017 |
| traders | 0.750 | 0.530 | 0.220 | 0.700 | 0.553 | 0.147 |
| justice | 0.421 | 0.370 | 0.051 | 0.544 | 0.617 | 0.073 |
| govt | 0.300 | 0.365 | 0.065 | 0.600 | 0.620 | 0.020 |
| business | 0.450 | 0.398 | 0.052 | 0.550 | 0.600 | 0.050 |
| maal | 0.370 | 0.348 | 0.022 | 0.497 | 0.631 | 0.134 |

---

### 6. How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

*DimABSA v2 (Contrastive-Aware) — September 15, 2026*
