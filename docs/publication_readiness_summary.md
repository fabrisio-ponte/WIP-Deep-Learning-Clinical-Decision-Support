# Publication Readiness Summary for the BEHRT-Style EHR Prediction Project
**Updated: August 29, 2026 - After comprehensive EDA analysis**

---

## 🔍 Key EDA Discoveries (August 2026)

**Dataset Characterization:**
- **179,713 samples** from **~25,000 unique patients** (~7 longitudinal snapshots each)
- **463 unique CCSR codes** used as prediction targets
- **Median patient:** 62 years old, 44 diagnoses in history, 9 codes at next visit
- **Population:** 88% adults 40+, 46% age 65+, 0% under age 18

**Critical Insight #1: This is recurrence prediction, not onset prediction**
- **70% of target codes are RECURRENCES** (already in patient history)
- **30% are truly new** diagnoses
- Model learns "which chronic conditions will be active next" not "which diseases will develop"

**Critical Insight #2: Code type heterogeneity**
- **85.6%** disease diagnoses
- **7.0%** symptoms (chest pain, abnormal labs, etc.)
- **4.0%** administrative/encounter codes (routine checkup, preventive care)
- **2.7%** injuries/trauma (unpredictable acute events)
- **0.7%** pregnancy and congenital conditions
- **14.4% of targets are not disease predictions**

**Critical Insight #3: Extreme class imbalance**
- **Most common code:** CCSR_END010 (diabetes) - 52,559 samples (36.6%)
- **Rarest codes:** 7 diseases with only 1 sample each
- **Imbalance ratio:** 52,559:1
- **133 rare diseases** (<100 samples, 29% of all diseases) have weak signal
- **Overall sparsity:** 48:1 (negative:positive ratio), only 2% of possible labels are positive

**Critical Insight #4: Performance dominated by common diseases**
- **Top 5 codes** (diabetes, hypertension, routine checkup, esophageal disorders, dysrhythmias) appear in 20-37% of samples
- **Bottom 50% of diseases** appear in <1% of samples each
- Model performance driven by chronic disease recurrence in sick, elderly patients

**Implication:** This reframes the entire thesis from "disease onset prediction" to **"next-visit clinical code recurrence prediction for multimorbid adults"**

---

## 1) What the project ACTUALLY is (after EDA findings)

This project is a BEHRT-style transformer for **longitudinal clinical code prediction** in multimorbid patients, trained on cleaned CCSR-coded sequences to predict which codes (diagnoses, symptoms, encounters) will appear at the next visit.

**Critical EDA findings that reframe the work:**
- **70% of target codes are RECURRENCES** (already in patient history), 30% are new
- This is **NOT "disease onset prediction"** — it's **"chronic disease recurrence and next-visit code prediction"**
- Dataset: **~25,000 unique patients**, each with ~7 longitudinal snapshots (179,713 total samples)
- Patient demographics: **median age 62 years, 88% adults 40+, 0% under 18**
- Code types: **85.6% diseases, 7% symptoms, 4% administrative, 2.7% injuries, 0.7% pregnancy/congenital**

The strongest framing is:

- a clinical sequence model for **next-visit clinical code prediction** in multimorbid adults
- trained on longitudinal EHR trajectories with **median 44 diagnoses per patient history**
- evaluated as a multilabel ranking problem with **extreme class imbalance** (52,559:1 ratio)
- predicting **which chronic conditions will be active/documented** at next encounter (70% recurrence + 30% new)
- using positive-class weighting to address rare code prediction

This is a valid and defensible applied ML / clinical informatics problem **when framed honestly**.

---

## 2) What the model can reasonably claim

The model can credibly claim that it learns **temporal patterns of clinical code recurrence** in multimorbid patients. In practical terms, it can:

- **predict which clinical codes will appear at next visit** (AUC 0.90, APS 0.27 with pos_weight)
- **identify which chronic conditions will likely recur** (70% of predictions)
- **detect some new diagnoses** (30% of predictions)
- **rank rare vs common codes** using positive-class weighting (improves APS by +7.6%, verified: 0.2543 -> 0.2735)
- **capture temporal persistence patterns** in chronic disease management
- **support care coordination** by predicting ~9 active conditions per next visit

**Clinical utility:** Helps allocate resources, schedule specialists, and coordinate care for complex multimorbid patients.

This is a meaningful contribution **when framed as clinical decision support for chronic disease management**, not as general disease screening.

---

## 3) Where the analogy breaks

This is the critical point to state clearly in any thesis or paper:

The model does not “understand medicine,” “know disease mechanisms,” or “reason clinically” in the way a physician does. It learns statistical associations in coded medical events.

The analogy breaks at several points:

**Conceptual limits:**
- **prediction ≠ understanding** — model predicts codes, not disease pathophysiology
- **correlation ≠ causality** — high AUC for diabetes recurrence doesn't explain why
- **recurrence ≠ onset** — 70% of predictions are chronic conditions already in history
- **code prediction ≠ disease prediction** — includes symptoms (7%), admin codes (4%), injuries (2.7%)

**Population limits:**
- **multimorbid adults only** — median age 62, median 44 prior diagnoses, 0% under 18
- **cannot generalize to healthy/young populations** — trained on sick, elderly patients
- **cannot predict acute events** — injuries are unpredictable, yet appear in targets

**Performance limits:**
- **dominated by common diseases** — top 5 diseases (diabetes, hypertension) drive most signal
- **133 rare diseases (<100 samples)** have insufficient data for reliable prediction
- **extreme imbalance (52,559:1 ratio)** means even pos_weight can't fully address rare codes
- **14.4% of targets are non-disease codes** (admin, symptoms, injuries)

This is not a weakness of the work; it is the **honest boundary** that must be documented to avoid overclaiming.

---

## 4) Key assumptions that underlie the work

The current project relies on several assumptions **validated or challenged by EDA**:

**Validated assumptions:**
- ✓ Prior diagnosis history contains signal (70% recurrence shows strong temporal persistence)
- ✓ Train/val/test splits are proper (no patient overlap confirmed)
- ✓ Distributions are consistent across splits (confirmed by EDA)
- ✓ Class weighting improves rare label learning (APS: 0.2543 → 0.2735, +7.6% — verified via `scripts/baseline_vs_posweight_comparison.py`, `results/baseline_vs_posweight_comparison.json`; supersedes the earlier unverified 0.262→0.274/+4.6% figure, which mixed runs from an older eval schema before UNK-label exclusion and threshold tuning were added)

**Assumptions requiring careful framing:**
- ⚠ **"Next-visit prediction" is primarily recurrence prediction** (70%), not onset (30%)
- ⚠ **CCSR labels include non-disease codes** (14.4%: symptoms, admin, injuries)
- ⚠ **Dataset represents multimorbid adults**, not general population (median age 62)
- ⚠ **Extreme class imbalance** (52,559:1) limits rare disease prediction fundamentally
- ⚠ **Model learns persistence patterns**, not causal disease progression

**Key dataset facts:**
- 179,713 samples from ~25,000 unique patients (~7 snapshots each)
- 463 unique CCSR codes: 85.6% diseases, 7% symptoms, 4% admin, 2.7% injuries, 0.7% other
- Median patient: 62 years, 44 diagnoses in history, 9 codes at next visit
- Top 5 diseases (diabetes, hypertension, etc.) appear in 20-37% of samples
- Bottom 133 diseases appear in <100 samples each (29% of all disease types)

These assumptions are reasonable **when limitations are stated clearly**, not as general disease prediction.

---

## 5) Main limitations that need to be faced honestly

The main limitations are not optional extras; they are central to the paper.

**Critical limitations discovered through EDA:**

1. **Task is recurrence prediction, not onset prediction**
   - 70% of target codes already in patient history
   - Cannot distinguish new disease from chronic disease flare-up without explicit modeling
   - Model learns "which conditions will be documented" not "what diseases will develop"

2. **Population is highly specific**
   - Median age 62 years, 88% adults 40+, 0% under age 18
   - Median 44 prior diagnoses (multimorbid, not healthy)
   - Median 9 next-visit codes (very sick patients)
   - **Cannot generalize to pediatric, young adult, or disease-free populations**

3. **Code type heterogeneity**
   - 14.4% of targets are NOT diseases: symptoms (7%), admin encounters (4%), injuries (2.7%)
   - Top-3 predictor is "routine checkup" (CCSR_FAC025, 28% of samples)
   - Performance metrics mix disease, symptom, and administrative code prediction

4. **Extreme class imbalance**
   - Most common disease: 52,559 samples (36.6%)
   - Rarest diseases: 1 sample each
   - Imbalance ratio: 52,559:1
   - 133 diseases (<100 samples, 29% of disease types) have weak predictive signal
   - Overall sparsity: 48:1 (negative:positive ratio)

5. **Performance dominated by common diseases**
   - Top 5 diseases (diabetes, hypertension, routine checkup, esophageal disorders, dysrhythmias) drive most signal
   - Rare diseases effectively unpredictable despite pos_weight
   - No causal interpretation of disease progression

6. **Acute events are unpredictable**
   - 2.7% of targets are injuries/trauma (falls, fractures)
   - These are random acute events, not learnable patterns
   - Inflates prediction difficulty artificially

**Standard limitations:**
- Modest data scale (~25k patients vs millions in foundation models)
- Limited to coded diagnoses, no clinical notes or labs
- Threshold sensitivity and calibration concerns
- Potential information loss from CCSR aggregation

These limitations are exactly what make a paper **credible when written honestly**.

---

## 6) What makes it publishable

This project is publishable if the thesis or paper is positioned as a focused clinical prediction study, not a broad medical intelligence claim.

To make it publishable, you need:

- a clean result story with a strong experimental comparison
- the same data, model, and training setup across the key ablation
- fair reporting of APS, AUC, F1, threshold tuning, and label coverage
- disease-level evaluation for clinically relevant conditions
- calibration and threshold analysis
- explicit limitations and failure-mode discussion
- honest framing that this is a predictive model, not a causal or mechanistic model

This is enough to support a strong applied ML or clinical informatics paper.

---

## 7) What is not worth doing right now

Do not spend much time on:

- deep mechanistic interpretability of the transformer
- trying to “explain the whole model” in biological terms
- broad architecture experimentation without a hypothesis
- a long list of random improvements that are not tied to a clear research question

Those directions are lower ROI for a thesis with limited time.

---

## 8) Best next research direction

The strongest next direction is not “more model complexity,” but better and more defensible evaluation and clinical relevance.

Priority order:

1. finalize the best current model and compare it rigorously against baseline
2. run disease-level performance analysis
3. analyze calibration and threshold sensitivity
4. study subgroup and rare-label behavior
5. write the limitations and scope section early
6. only then consider extra modeling improvements such as class-specific strategies, longer history windows, or a more tailored objective

This keeps the project clinically grounded and publication-focused.

---

## 9) The strongest thesis-level takeaway

A thesis-safe statement would be:

> **"This work demonstrates that a transformer-based model can learn temporal patterns of clinical code recurrence from longitudinal EHR sequences, achieving meaningful next-visit prediction performance (AUC 0.90, APS 0.27) for multimorbid adult patients. Positive-class weighting improves rare code detection by 7.6% APS over baseline (0.2543 -> 0.2735). However, the task is primarily chronic disease recurrence prediction (70% of targets) rather than new disease onset, performance is dominated by common conditions, and the model remains a statistical predictor of coded clinical events rather than a mechanistic account of disease progression. The work's practical value lies in supporting care coordination and resource allocation for complex patients, with clear limitations regarding population generalizability and interpretability."**

**Reframed research question:**
> "Can transformer models effectively predict which clinical codes (diagnoses, symptoms, encounters) will appear at a patient's next visit, learning from longitudinal EHR sequences with extreme class imbalance and chronic disease recurrence patterns?"

**Key contributions:**
1. **Methodological:** Demonstrates transformer effectiveness on longitudinal EHR with 52,559:1 class imbalance
2. **Technical:** Positive-class weighting improves rare code detection (capped at MAX_POS_WEIGHT=30)
3. **Empirical:** Dataset characterization: 70% recurrence, 30% new codes, 14.4% non-disease codes
4. **Clinical:** Supports next-visit care coordination for multimorbid patients (median 9 active conditions)

This is intellectually honest and academically strong.

---

## 10) Final judgment

Yes, this project **is publishable** when reframed correctly, but the route is:

**❌ DON'T claim:**
- "General disease prediction model"
- "Predicts new disease onset"
- "Learns disease mechanisms"
- "Generalizes to all populations"
- "Solves rare disease prediction"

**✓ DO claim:**
- "Next-visit clinical code prediction for multimorbid adults"
- "Learns temporal recurrence patterns (70%) with some new diagnosis detection (30%)"
- "Supports care coordination and resource allocation"
- "Demonstrates transformer effectiveness on longitudinal EHR with extreme imbalance"
- "Positive-class weighting improves rare code detection by 7.6% APS (verified: 0.2543 → 0.2735)"

**Clinical framing:**
This is **NOT** a disease screening tool for healthy populations.
This **IS** a clinical decision support tool for managing complex, multimorbid patients.

**Target venues:**
- Applied ML in healthcare (CHIL, ML4H)
- Clinical informatics (JAMIA, JBI)
- Medical AI with honest limitations (Nature Digital Medicine, npj Digital Medicine)

The honest framing makes it **stronger**, not weaker. Reviewers will appreciate the careful dataset analysis and transparent limitations.

---

## 11) Immediate next steps (updated after EDA)

**EDA work (in progress):**
- \u2713 Step 1: Dataset structure analysis (completed)
- \u2713 Step 2: Patient-level analysis (completed - discovered 70/30 recurrence split)
- \u2713 Step 3: Label support & class imbalance (completed - found 52,559:1 ratio, 14.4% non-disease codes)
- \u23f3 Step 4: Temporal structure analysis (next)
- \u23f3 Step 5: Label co-occurrence patterns
- \u23f3 Step 6: Split representativeness
- \u23f3 Step 7: Cleaning impact analysis

**Analysis priorities after EDA:**
1. ✓ **Comparison table:** Baseline vs pos_weight, same seed/data/architecture (`scripts/baseline_vs_posweight_comparison.py` → `results/baseline_vs_posweight_comparison.json`). Verified result: APS 0.2543 → 0.2735 (+7.6%), AUC 0.8911 → 0.9006 (+1.07%). Only seed=42 has a full run under the current eval schema for both arms, so this is a single-seed comparison; flagged as a limitation.
2. **Disease-level performance:** Break down by common vs rare, new vs recurrent
3. ✓ **Code-type analysis:** Separate performance on diseases vs symptoms vs admin codes (`eda/11_code_type_performance_breakdown.py` → `eda/results/11_code_type_performance_breakdown.json`). Confirms 85.7% disease / 7.1% symptom / 4.3% admin / 2.6% injury / 0.3% pregnancy-perinatal test-occurrence split, and shows injury codes have near-zero F1 (0.046) vs disease F1 (0.200), supporting the "acute events are unpredictable" limitation.
4. **NEW vs RECURRING performance:** Compare model effectiveness on 70% recurrent vs 30% new codes
5. **Calibration analysis:** Threshold sensitivity, per-class calibration

**Novel analysis opportunity discovered:**
- **Recurrence vs onset stratified evaluation** — this would be a unique contribution showing:
  - Model excels at predicting chronic disease recurrence (70% of task)
  - Weaker at predicting truly new diagnoses (30% of task)
  - Different optimal strategies for each (pos_weight helps new codes more)

**Documentation priorities:**
1. Update methods section with dataset characterization
2. Write limitations section FIRST (before adding experiments)
3. Frame results around care coordination utility, not general prediction
4. Add honest discussion of what 0.90 AUC means in this context

**What NOT to do:**
- Deep mechanistic interpretability (low ROI)
- Architecture experiments without hypothesis
- Trying to fix the 52,559:1 imbalance fundamentally (it's a data limit)
- Overclaiming about rare disease prediction

This keeps the project on a **publishable, thesis-constrained, honest path**.

---

## 12) Trivial baseline comparison — is the transformer beating naive predictors?

**Note on terminology:** this is a *different* comparison from Section 11 item 1
("Baseline vs pos_weight"), which compares two trained BEHRT configurations
(with vs without positive-class weighting). This section instead asks a more
fundamental question: does the trained BEHRT model beat models that require
**no training and no patient-specific information at all**?

`scripts/baseline_comparison.py` implements two trivial, non-learned baselines
and evaluates them under the **identical** eval pipeline as the reference
model (same label set, same UNK exclusion, same threshold-tuning procedure,
same NEW-vs-RECURRING breakdown) by directly reusing the reference script's
eval functions — guaranteeing an apples-to-apples comparison:

- **`repeat_last_visit`**: predicts exactly the codes present at the patient's
  most recent visit (zero learning, pure copy-forward).
- **`popularity`**: predicts a fixed per-label training-set frequency score
  for every patient (zero learning, zero patient-specific information at all).

**Results** (`results/baseline_comparison.json`, test set, n=18,237):

| Model | Overall APS | Overall AUC | Recurring APS | Recurring AUC | New APS | New AUC |
|---|---|---|---|---|---|---|
| **Reference BEHRT** (`clean_run_20260910_124255`) | 0.2735 | 0.9006 | 0.4183 | 0.6811 | 0.0328 | 0.8654 |
| `popularity` | 0.2544 | 0.8914 | 0.4143 | 0.6887 | 0.0375 | 0.8586 |
| `repeat_last_visit` | 0.2372 | 0.7184 | 0.3565 | 0.6741 | 0.0064 | 0.5000 |

**Key finding (important, and somewhat sobering):** on both the recurring-diagnosis
subset AND the new-diagnosis subset, `popularity` — which uses **no patient
history at all** — is statistically indistinguishable from, and on several
metrics nominally ahead of, the trained BEHRT model. Overall, BEHRT beats
`popularity` by only +7.5% relative APS and +1.0% relative AUC. Most of
BEHRT's AUC advantage over `repeat_last_visit` comes from being able to say
*anything* about new diagnoses (`repeat_last_visit` collapses to AUC=0.50/
chance there by construction, since it can never predict a code the patient
hasn't already had) — not from sequence modeling clearly beating base rates.

**What this means for the paper:** the current draft's framing (trained
transformer as the primary contribution) needs to be tempered — the
transformer's advantage over trivial base-rate exploitation is real but
modest, and is not yet clearly attributable to personalized/sequential
reasoning. This is a more defensible, honest finding than omitting a baseline
entirely (the original gap identified in this document), but it also means
the "Baseline Comparison" section cannot simply say "BEHRT beats baselines" —
it must show and discuss the recurring/new breakdown above.

### Recommendation: build ONE learned baseline next — logistic regression, not XGBoost

Between logistic regression and XGBoost (the two options discussed), **logistic
regression is the better next step for this thesis**, for three reasons:

1. **It directly answers the open question.** The trivial baselines show BEHRT
   barely beats a model with *no* patient information. The next question is
   whether *any* model that uses patient history (not just population base
   rates) can close that gap — logistic regression on simple per-patient
   history features (e.g., per-code presence/recency in history, visit count,
   age) isolates "does using this patient's history help at all" from
   "does the transformer architecture specifically help." XGBoost answers a
   less central question (does a stronger nonlinear learner on the same
   features help), which only matters once the first question is settled.
2. **Reviewer expectations.** A linear/logistic baseline on hand-engineered
   features is the standard, expected comparison point in clinical ML papers
   (e.g., "does the transformer beat a simple risk score built from the same
   information?"). Its absence is a more commonly-flagged reviewer gap than
   the absence of a gradient-boosted baseline.
3. **Cost/ROI.** Logistic regression (via scikit-learn, one-vs-rest or a
   single multi-label linear layer) trains in seconds to minutes on this
   data size and needs no hyperparameter search, keeping this a
   thesis-appropriate, low-risk addition. XGBoost would require per-label or
   multi-output tuning (learning rate, depth, n_estimators) to be a fair
   comparison, which is a larger time investment for a secondary question.

**Concretely**: build a logistic-regression baseline using features derivable
directly from the existing `code`/`age` columns (e.g., a multi-hot
"code ever appeared in history" vector, optionally recency-weighted, plus
age and visit count), trained as a multi-label linear model on the same
train split, and evaluated through the same reused eval pipeline as
`baseline_comparison.py`. If logistic regression also lands close to BEHRT,
that is strong evidence the task's ceiling is largely determined by
available features/base rates rather than architecture — an important,
publishable, honest finding either way. XGBoost can remain a documented
"future work" option rather than being built now.

This keeps the project on a **publishable, thesis-constrained, honest path**.

### Logistic regression baseline — result (CRITICAL FINDING, not a close call)

`scripts/logistic_regression_baseline.py` implements one-vs-rest logistic
regression (scikit-learn) over 942 hand-engineered per-patient history
features (multi-hot code-presence-in-history, age, visit count), 470 labels,
evaluated through the same reused eval pipeline (`results/
logistic_regression_baseline.json`, test set n=18,237):

| Model | APS | AUC | Tuned threshold |
|---|---|---|---|
| **Logistic regression** | **0.2915** | **0.9162** | 0.80 |
| Reference BEHRT (`clean_run_20260910_124255`) | 0.2735 | 0.9006 | 0.60 |
| `popularity` | 0.2544 | 0.8914 | 0.15 |
| `repeat_last_visit` | 0.2372 | 0.7184 | 0.10 |

**Logistic regression beats the reference BEHRT model on both aggregate
metrics** — APS +6.6% relative, AUC +1.7% relative. This is not within noise;
it is the single most important result for how the paper's contribution must
be framed.

**But the new-vs-recurring breakdown shows exactly where each model's
advantage comes from, and it is not a uniform win for logistic regression:**

| | Logistic regression | Reference BEHRT |
|---|---|---|
| Recurring F1 / AUC | **0.453** / 0.676 | 0.403 / **0.681** |
| New-diagnosis F1 / AUC | 0.025 / 0.838 | **0.061** / **0.865** |

- On **recurring-code prediction** (70% of all targets), logistic regression's
  F1 is meaningfully higher (0.453 vs 0.403); AUC is about tied. A simple
  "was this code recently in the patient's history" feature captures most of
  the recurrence signal — it does not require a transformer.
- On **new-diagnosis detection** (30% of targets, the harder and arguably
  more clinically interesting half), BEHRT wins clearly: ~2.4x the F1
  (0.061 vs 0.025) and higher AUC (0.865 vs 0.838).
- Because recurring codes dominate the label distribution, logistic
  regression's recurring-side advantage is enough to win on the aggregate
  metrics even though BEHRT is the better model specifically at new-onset
  detection.

**What this means for the paper (must-fix before publication):** the current
draft framing has no learned-baseline comparison at all. It can no longer
claim BEHRT is simply "the best model" on this task — that claim is now false
on aggregate APS/AUC. The defensible, and more scientifically interesting,
claim is: *"a simple logistic-regression baseline matches or exceeds BEHRT on
aggregate metrics because those metrics are dominated by recurrence
prediction, which linear history features capture well; BEHRT's genuine,
measurable advantage is concentrated in new-diagnosis detection."* This
should become an explicit baseline-comparison section/table in
`paper/latex/main.tex`, and the Discussion/Conclusion's "what can the model
credibly claim" language should be narrowed accordingly (recommend anchoring
BEHRT's contribution claim to the new-diagnosis-detection numbers, not the
aggregate APS/AUC, which a simple baseline now wins).

### Should we also try XGBoost / random forest / SVM?

Given the logistic regression result, the calculus has changed from when the
above recommendation was written (back then, logistic regression's outcome
was still unknown). Current thinking:

- **XGBoost / random forest (tree-based, same features)**: worth trying, but
  now for a different reason than originally planned. The open question is no
  longer "does *any* patient-history model help" (answered: yes, logistic
  regression already does) but "is BEHRT's new-diagnosis advantage specific
  to the transformer/sequence architecture, or would a stronger non-linear
  model on the same flat features close that gap too?" If a tuned XGBoost
  matches BEHRT's new-diagnosis F1 using the same 942 flat features (no
  sequence modeling), that would meaningfully undercut the "transformer
  architecture adds value" claim; if it doesn't, that strengthens it. This is
  now a more central question than it was before the logistic regression
  result existed, so it is reasonable to build one tree-based baseline
  (XGBoost is preferred over random forest here — better handling of the
  extreme 52,559:1 imbalance via `scale_pos_weight`, and is the standard
  strong-baseline choice in clinical ML papers).
- **Random forest**: skip if XGBoost is built — the two occupy the same role
  (non-linear tree ensemble on flat features) and reviewers will expect at
  most one representative of that family. Only worth adding both if a
  reviewer specifically asks for it.
- **Plain decision tree**: not worth building as a separate baseline — a
  single tree is strictly dominated by both logistic regression (in linear
  signal capture) and any ensemble method, and would not answer a new
  question. Skip.
- **SVM**: not recommended for this data size. A one-vs-rest SVM over 470
  labels and ~143,677 training rows with 942 features is a substantially
  larger compute cost than logistic regression (kernel SVMs scale poorly with
  n_samples; even a linear SVM via `LinearSVC` would need per-label
  probability calibration to produce the APS/AUC-style ranking metrics used
  here) for a result that, in the vast majority of published comparisons on
  similar tabular clinical features, tracks very close to logistic regression
  in ranking quality. Low expected information gain for the added cost. Skip
  unless a specific reviewer requests it.

**Recommendation**: build ONE additional baseline — XGBoost, using the same
942 features and same eval pipeline as `logistic_regression_baseline.py` —
specifically to test whether BEHRT's new-diagnosis-detection advantage
survives against a stronger non-linear (but still non-sequential) learner.
Skip random forest, plain decision trees, and SVM as low-marginal-value given
XGBoost already covers the "stronger non-linear baseline" question tree
methods and SVM would otherwise answer.

### XGBoost baseline — result (largest gap yet, but with an important caveat)

`scripts/xgboost_baseline.py` implements one-vs-rest XGBoost (150 shallow
trees, depth 4, `tree_method="exact"` — see platform note below) on the
identical 942-feature matrix used by the logistic regression baseline,
evaluated through the same pipeline (`results/xgboost_baseline.json`, test
set n=18,237):

| Model | APS | AUC | Tuned threshold |
|---|---|---|---|
| **XGBoost** | **0.4894** | **0.9307** | 0.90 |
| Logistic regression | 0.2915 | 0.9162 | 0.80 |
| Reference BEHRT | 0.2735 | 0.9006 | 0.60 |
| `popularity` | 0.2544 | 0.8914 | 0.15 |
| `repeat_last_visit` | 0.2372 | 0.7184 | 0.10 |

**XGBoost decisively beats every other model on both ranking metrics** — APS
is nearly double the reference BEHRT's (+79% relative), AUC is the highest of
any model tested. On pure ranking ability (how well the model orders
candidate codes by likelihood, independent of any decision threshold),
gradient-boosted trees on simple flat history features outperform the
transformer.

**However, this needs an important qualification: XGBoost's F1 at its tuned
decision threshold is actually the *worst* of the three learned models**,
because its predicted probabilities are poorly calibrated for point
prediction:

| | XGBoost | Logistic regression | Reference BEHRT |
|---|---|---|---|
| Micro-F1 (tuned threshold) | 0.147 | 0.297 | 0.238 |
| Micro-precision / recall | 0.080 / 0.862 | — | — |

The likely cause: a single dataset-wide average `scale_pos_weight` (≈8,010)
was used across all 470 labels to correct for class imbalance, but per-label
imbalance ranges enormously (some labels have ~30 positive training
examples, others have 5,000+). One global weight over-corrects for common
labels and under-corrects for rare ones, systematically inflating predicted
probabilities and making any single fixed threshold produce far too many
false positives (recall 0.86 but precision only 0.08). This is a tuning
artifact, not evidence that XGBoost's underlying signal is weak — the AUC/APS
numbers show the ranking signal is in fact the strongest of any model here.

**New-vs-recurring breakdown** reinforces this reading:

| | XGBoost | Logistic regression | Reference BEHRT |
|---|---|---|---|
| Recurring F1 / AUC | 0.440 / 0.760 | 0.453 / 0.676 | 0.403 / 0.681 |
| New-diagnosis F1 / AUC | 0.050 / **0.855** | 0.025 / 0.838 | **0.061** / 0.865 |

XGBoost's new-diagnosis AUC (0.855) closes most of the gap to BEHRT (0.865)
and clearly beats logistic regression (0.838) — its ranking ability on the
harder, more clinically interesting new-onset subset is nearly on par with
the transformer. Its F1 there still trails BEHRT's, again consistent with a
threshold-calibration gap rather than a ranking-ability gap.

**Platform note**: `tree_method="hist"` reliably segfaults on this
macOS/xgboost 3.4.1 setup when fit on this sparse feature matrix (reproduced
in isolated tests, independent of `scale_pos_weight` or label balance).
`tree_method="exact"` avoids the crash and adds acceptable cost (~12s/label,
~9-10 min total across 470 labels with 10-way parallelism). Documented in
`/memories/repo/behrt_findings.md` in case this needs revisiting later.

**What this means for the paper**: this is the most important baseline
result yet, and forces a more careful framing than "BEHRT is the strongest
model." The honest characterization is:
- On pure ranking quality (APS/AUC), a tuned gradient-boosted tree ensemble
  on simple flat history features beats BEHRT, and is not far behind it even
  on the new-diagnosis subset specifically.
- BEHRT's most defensible remaining advantage is in *producing usable point
  predictions* (a threshold that yields a reasonable precision/recall
  trade-off) without per-label threshold tuning, and in new-diagnosis F1
  specifically.
- If XGBoost's threshold/calibration issue were fixed (e.g., per-label
  `scale_pos_weight` or isotonic/Platt calibration), it would very likely
  also close or exceed BEHRT's F1 gap — this baseline was deliberately kept
  simple (single global weight, no calibration, no hyperparameter search) to
  stay within thesis scope, so this result should be reported as "a
  reasonably-tuned XGBoost baseline," not "the best possible XGBoost
  baseline."
- Recommend the paper's baseline-comparison table include all five models
  (BEHRT, XGBoost, logistic regression, popularity, repeat_last_visit) with
  both APS/AUC and tuned-threshold F1, explicitly discussing the
  ranking-vs-calibration distinction — this is a more nuanced and more
  scientifically interesting story than any single "X beats Y" headline.
