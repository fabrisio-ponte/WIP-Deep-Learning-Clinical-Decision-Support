# Recommended Reading: EHR Transformers, Diagnosis Prediction, Calibration, and Interpretability

This is a compact reading list of widely cited papers that are directly relevant to the themes in this project: longitudinal EHR modeling, next-visit diagnosis prediction, transformer-style representation learning, class imbalance, calibration, and interpretability.

The list is organized for practical reading rather than completeness. The first section contains the highest-priority anchors.

## 1. Highest-Priority Anchor Papers

1. Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
Why it matters: the base transformer architecture used indirectly by BEHRT.

2. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.
Why it matters: the pretraining recipe BEHRT adapts to longitudinal medical sequences.

3. Li et al. (2020). BEHRT: Transformer for Electronic Health Records. Scientific Reports.
Why it matters: the core model lineage for this thesis.

4. Rasmy et al. (2021). Med-BERT: Pretrained Contextualized Embeddings on Large-Scale Structured Electronic Health Records for Disease Prediction. npj Digital Medicine.
Why it matters: one of the strongest and most cited follow-on papers for transformer-style EHR pretraining.

5. Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
Why it matters: foundational calibration paper for understanding why ranking metrics and decision-threshold metrics can diverge.

6. Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks. ICML.
Why it matters: the standard Integrated Gradients paper.

7. Jain and Wallace (2019). Attention Is Not Explanation. NAACL.
Why it matters: the main cautionary paper for interpreting attention weights.

## 2. Classic Clinical Sequence Modeling / Diagnosis Prediction

1. Choi et al. (2016). RETAIN: An Interpretable Predictive Model for Healthcare Using Reverse Time Attention Mechanism. NeurIPS.
Why it matters: classic sequential EHR baseline and one of the main pre-transformer healthcare sequence papers.

2. Nguyen et al. (2017). Deepr: A Convolutional Net for Medical Records. IEEE Journal of Biomedical and Health Informatics.
Why it matters: early deep learning approach to structured clinical trajectories.

3. Choi et al. (2016). Doctor AI: Predicting Clinical Events via Recurrent Neural Networks. MLHC.
Why it matters: foundational next-visit prediction framing close to the spirit of this project.

4. Baytas et al. (2017). Patient Subtyping via Time-Aware LSTM Networks. KDD.
Why it matters: classic treatment of irregular time gaps in clinical sequence modeling.

5. Rajkomar et al. (2018). Scalable and Accurate Deep Learning with Electronic Health Records. npj Digital Medicine.
Why it matters: large-scale benchmark paper showing how much predictive signal is available in raw EHR data.

## 3. Structured EHR Transformers and Representation Learning

1. Shang et al. (2019). Pre-training of Graph Augmented Transformers for Medication Recommendation. IJCAI.
Why it matters: not diagnosis prediction directly, but important for graph-aware medical transformers.

2. Shang et al. (2021). G-BERT: Pre-training of Text and Structured Data for Medical Representation Learning. IEEE Transactions on Knowledge and Data Engineering.
Why it matters: combines structured EHR-style information with pretraining ideas similar in spirit to BEHRT.

3. Pang et al. (2021). CEHR-BERT: Incorporating Temporal Information from Structured EHR Data to Improve Prediction Tasks.
Why it matters: useful if you want other EHR-BERT variants to compare against BEHRT and Med-BERT.

## 4. Imbalance, Long-Tail Learning, and Thresholding

1. Saito and Rehmsmeier (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE.
Why it matters: key paper for explaining why APS/PR behavior matters more than ROC alone in heavily imbalanced tasks.

2. Wu et al. (2020). Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets. ECCV.
Why it matters: directly relevant to rare-label handling in multi-label settings.

3. Menon et al. (2021). Long-Tail Learning via Logit Adjustment. ICLR.
Why it matters: strong modern treatment of long-tail bias at the logit level.

4. Cui et al. (2019). Class-Balanced Loss Based on Effective Number of Samples. CVPR.
Why it matters: very widely cited weighting approach; not clinical-specific, but often adapted.

5. Ridnik et al. (2021). Asymmetric Loss for Multi-Label Classification. ICCV.
Why it matters: useful if you later want stronger multi-label loss baselines than simple BCE with pos-weight.

## 5. Calibration, Decision Thresholds, and Clinical Reliability

1. Niculescu-Mizil and Caruana (2005). Predicting Good Probabilities with Supervised Learning. ICML.
Why it matters: classic calibration reference before the deep-learning era.

2. Van Calster et al. (2019). Calibration: The Achilles Heel of Predictive Analytics. BMC Medicine.
Why it matters: excellent clinical-model calibration overview, very readable for discussion sections.

3. Ovadia et al. (2019). Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. NeurIPS.
Why it matters: highly useful for reliability arguments once a clinical model leaves the training distribution.

4. Kompa et al. (2021). Second Opinion Needed: Communicating Uncertainty in Medical Machine Learning. npj Digital Medicine.
Why it matters: strong reference for uncertainty, deployment caution, and clinical interpretability limits.

## 6. Interpretability and Attribution for Clinical Models

1. Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks. ICML.
Why it matters: Integrated Gradients foundation.

2. Jain and Wallace (2019). Attention Is Not Explanation. NAACL.
Why it matters: strong caution against overclaiming from attention maps.

3. Abnar and Zuidema (2020). Quantifying Attention Flow in Transformers. ACL.
Why it matters: attention rollout reference used in this project.

4. Ribeiro et al. (2016). Why Should I Trust You? Explaining the Predictions of Any Classifier. KDD.
Why it matters: LIME; broad interpretability background.

5. Lundberg and Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
Why it matters: SHAP; useful comparison point even if not used here.

## 7. Best Search Terms to Keep Following the Literature

Use these phrases in Google Scholar, Semantic Scholar, arXiv, or Papers With Code:

1. "EHR transformer diagnosis prediction"
2. "next visit diagnosis prediction electronic health records"
3. "clinical code prediction transformer"
4. "structured EHR pretraining"
5. "multi-label long-tail calibration healthcare"
6. "new diagnosis detection EHR"
7. "clinical model calibration threshold selection"

## 8. Suggested Reading Order for This Project

1. Attention Is All You Need
2. BERT
3. BEHRT
4. Med-BERT
5. RETAIN
6. Doctor AI
7. Guo et al. on calibration
8. Saito and Rehmsmeier on PR vs ROC
9. Wu et al. on distribution-balanced loss
10. Menon et al. on logit adjustment
11. Jain and Wallace
12. Sundararajan et al.
13. Abnar and Zuidema

## 9. Most Directly Useful Papers for This Thesis Writeup

If time is limited, prioritize these for discussion/future-work citations:

1. Li et al. (BEHRT)
2. Rasmy et al. (Med-BERT)
3. Choi et al. (RETAIN)
4. Guo et al. (calibration)
5. Saito and Rehmsmeier (PR vs ROC)
6. Wu et al. (distribution-balanced loss)
7. Menon et al. (logit adjustment)
8. Jain and Wallace (attention caution)
9. Sundararajan et al. (Integrated Gradients)
10. Abnar and Zuidema (attention rollout)

## 10. Practical Note

For exact up-to-date citation counts, use Google Scholar or Semantic Scholar directly. This file is intended as a curated reading map, not a citation-count registry.