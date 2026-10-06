# E-commerce Conversion Prioritization Lab

An interactive model-selection and threshold-review demo using the UCI Online Shoppers Purchasing Intention dataset. It shows how candidate ranking metrics and a fixed test-set threshold curve can inform a limited human review queue.

**[Open live demo](https://sumeetamin.github.io/ecommerce-conversion-prioritization/)**

On the held-out 2,466-session test set, the selected histogram gradient boosting model achieved 0.737 average precision and 0.930 ROC-AUC. At the validation-selected 0.45 threshold, precision was 0.689 and recall was 0.649. The majority-probability baseline AP was 0.155.

## What is measured

- Logistic regression and histogram gradient boosting are compared on validation average precision.
- The selected model is refit on train + validation. The held-out test split is scored once.
- Test metrics include average precision, ROC-AUC, Brier score, precision, recall and F1. The threshold was selected on validation by F1 and is then reported on the test set.
- The dashboard exposes precision, recall, sessions reviewed and conversions captured across score thresholds, plus aggregate visitor-type outcomes.

The features describe the session, including page-view duration and exit rates. The score is therefore an **end-of-session** prioritization signal. It cannot be used as evidence that an email, discount, or other intervention causes incremental conversion. No per-session predictions are published.

## Dataset and attribution

Sakar, C. & Kastro, Y. (2018). [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset). UCI Machine Learning Repository. DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). License: CC BY 4.0.

The published dataset contains 12,330 sessions and a binary `Revenue` target. This repository stores the reproducible runner and aggregate report; it does not commit the raw CSV or individual session scores.

## Reproduce

From the repository root:

```powershell
python -m pip install -r requirements.txt
python benchmark/run.py
```

The script downloads the UCI archive to the ignored `benchmark_data/` directory and writes `data/benchmark.json`.

## Limitations

- The random stratified split measures interpolation within this historical dataset, not temporal generalization to a future store or campaign.
- Average precision and ROC-AUC measure ranking; Brier score measures probability error. None imply business value without costs, benefits and a tested intervention.
- Visitor segments can have small sample sizes. The benchmark is not evidence of fairness, and the demographic fields needed for a fairness audit are not present.
- Score thresholds should be chosen on validation data and revalidated under the real operating capacity and action costs.

