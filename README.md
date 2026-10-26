# CDR-Based Cognitive Impairment Classification (OASIS-1 Dataset)

M.Sc. Statistics and Computing project, Banaras Hindu University (DST CIMS).

This project classifies **CDR-defined cognitive impairment** (CDR = 0 vs CDR > 0) from demographic, clinical and structural neuroimaging-derived variables in the OASIS-1 cross-sectional dataset, using statistical analysis and machine learning. It is a research and learning project. It is **not** a diagnostic model for Alzheimer's disease or any other condition.

## Dataset

- Source: Open Access Series of Imaging Studies (OASIS-1, cross-sectional MRI data). Download it from the official OASIS website and follow its terms of use. Official site: https://sites.wustl.edu/oasisbrains/
- **The raw data file is not included in this repository.**
- The original file has 436 sessions (rows) and 12 variables, from 416 unique subjects (20 subjects have a repeat session).
- CDR is available for 235 rows, and these 235 rows are 235 unique subjects (none of the repeat sessions have a CDR). These 235 rows form the modelling dataset.
- Rows without a CDR (n = 201) have a much lower mean age (26.8 years, range 18 to 58) than rows with a CDR (72.3 years, range 33 to 96), so the modelled sample is mainly older adults.
- Target: CDR = 0 (no cognitive impairment): n = 135 (57.45%). CDR > 0 (cognitive impairment): n = 100 (42.55%), made up of CDR 0.5 (n = 70), CDR 1 (n = 28) and CDR 2 (n = 2).

### Variables

| Role | Variables |
|---|---|
| Predictors | Age, M/F, Educ, SES, MMSE, eTIV, nWBV, ASF |
| Excluded | ID (identifier), CDR (defines the target, so it would be target leakage), Hand (all 436 values are "R", so no variation), Delay (95.4% missing overall, and missing in all 235 modelled rows) |

## Workflow (as implemented in the notebook)

1. Load and audit the data: shape, missing values, duplicate rows (0), descriptive statistics, CDR distribution.
2. Create a subject identifier from the ID and check repeated subjects.
3. Restrict to rows with a known CDR and build the binary target (CDR > 0).
4. Missing-data analysis: among the 235 modelled rows only SES has missing values (19 rows, 8.09%).
5. Leakage audit: CDR, ID and Subject_ID are not used as predictors.
6. Group comparison (CDR = 0 vs CDR > 0) with two-sided Mann-Whitney U tests.
7. Subject-level, stratified 80/20 train-test split (`train_test_split`, `random_state=42`).
8. Models: Logistic Regression, Random Forest and XGBoost (details below).
9. Evaluation on the held-out test set.
10. 5-fold stratified cross-validation.
11. Bootstrap 95% confidence intervals for the test-set ROC-AUC.
12. Feature-importance analysis.

### Split

Training: 188 subjects (CDR = 0: 108; CDR > 0: 80). Testing: 47 subjects (CDR = 0: 27; CDR > 0: 20). Subject overlap: 0. In the 235 labelled rows every subject appears once, so the subject-level split is equivalent to a row-level stratified split.

### Missing values and preprocessing

- Sex is encoded as F = 0, M = 1.
- SES (19 missing values) is imputed with the **median**, using `SimpleImputer(strategy="median")` inside each model pipeline, so the imputation values come from the training data only (and from the training folds only in cross-validation).
- Logistic Regression also standardises all features (`StandardScaler`) inside its pipeline. Random Forest and XGBoost use unscaled features.

### Models

| Model | Settings (as in the code) |
|---|---|
| Logistic Regression | `max_iter=1000`, `random_state=42`; all other settings are scikit-learn defaults (L2 penalty, C = 1.0, lbfgs solver); features standardised |
| Random Forest | `n_estimators=500`, `max_depth=None`, `min_samples_split=5`, `min_samples_leaf=2`, `class_weight="balanced"`, `random_state=42` |
| XGBoost | `n_estimators=300`, `max_depth=3`, `learning_rate=0.03`, `min_child_weight=3`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`, `objective="binary:logistic"`, `eval_metric="logloss"`, `random_state=42` |

The notebook computes `scale_pos_weight` (108/80 = 1.35) for the training set but does **not** pass it to the XGBoost model, so XGBoost was trained without class weighting. Predictions use the default 0.5 probability threshold. No hyperparameter search is included in the notebook.

## Results

### Group comparison (two-sided Mann-Whitney U tests)

| Variable | CDR = 0 mean (SD) | CDR > 0 mean (SD) | p-value |
|---|---|---|---|
| Age | 69.074 (13.876) | 76.760 (7.120) | < 0.001 |
| Educ | 3.444 (1.262) | 2.820 (1.298) | 0.0003 |
| SES | 2.331 (1.064) | 2.747 (1.167) | 0.0094 |
| MMSE | 29.096 (1.126) | 24.320 (4.165) | < 0.001 |
| eTIV | 1441.53 (152.36) | 1483.75 (168.32) | 0.0518 |
| nWBV | 0.7692 (0.0459) | 0.7220 (0.0358) | < 0.001 |
| ASF | 1.2307 (0.1266) | 1.1974 (0.1313) | 0.0516 |

SES statistics use the 216 rows with a recorded SES. Sex by group (counts): female 97 (CDR = 0) and 59 (CDR > 0); male 38 and 41. No significance test for sex was run in the notebook.

### Test-set performance (n = 47)

| Model | Accuracy | Precision | Sensitivity | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7872 | 0.7778 | 0.70 | 0.7368 | 0.9056 |
| Random Forest | 0.8085 | 0.7895 | 0.75 | 0.7692 | 0.9296 |
| XGBoost | 0.8085 | 0.7895 | 0.75 | 0.7692 | 0.9333 |

Confusion matrices (rows: actual CDR = 0, CDR > 0; columns: predicted CDR = 0, CDR > 0): Logistic Regression [[23, 4], [6, 14]]; Random Forest and XGBoost [[23, 4], [5, 15]].

### 5-fold stratified cross-validation

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, run on all 235 labelled rows (so it is not independent of the test set above).

| Model | Accuracy | Precision | Sensitivity | F1 | ROC-AUC (SD) |
|---|---|---|---|---|---|
| Logistic Regression | 0.8170 | 0.8237 | 0.73 | 0.7665 | 0.9030 (0.0348) |
| Random Forest | 0.8255 | 0.8293 | 0.77 | 0.7890 | 0.9063 (0.0339) |
| XGBoost | 0.8128 | 0.7982 | 0.76 | 0.7735 | 0.9074 (0.0304) |

Fold-wise ROC-AUC: Logistic Regression 0.9315, 0.8778, 0.9444, 0.8500, 0.9111; Random Forest 0.9167, 0.8741, 0.9389, 0.8593, 0.9426; XGBoost 0.9019, 0.8870, 0.9593, 0.8704, 0.9185.

### Bootstrap 95% confidence intervals for test-set ROC-AUC

5,000 resamples of the 47 test subjects with replacement (`random_state=42`); resamples containing only one class are skipped; percentile intervals. The models are not refitted in each resample, so the intervals reflect test-sample variability only.

| Model | Observed AUC | 95% CI |
|---|---|---|
| Logistic Regression | 0.9056 | [0.8084, 0.9732] |
| Random Forest | 0.9296 | [0.8485, 0.9827] |
| XGBoost | 0.9333 | [0.8549, 0.9852] |

### Feature importance

Random Forest uses impurity-based `feature_importances_`; XGBoost uses `feature_importances_` of the scikit-learn wrapper (the importance type is not set explicitly).

| Feature | Random Forest | XGBoost |
|---|---|---|
| MMSE | 0.3887 | 0.4390 |
| nWBV | 0.1836 | 0.1226 |
| Age | 0.1549 | 0.1007 |
| eTIV | 0.0840 | 0.0637 |
| ASF | 0.0839 | 0.0620 |
| Educ | 0.0517 | 0.0711 |
| SES | 0.0312 | 0.0663 |
| M/F | 0.0221 | 0.0745 |

Logistic Regression coefficients (standardised features, per 1 SD): MMSE -2.605, nWBV -0.749, Educ -0.198, Age -0.174, SES -0.024, ASF 0.066, M/F 0.133, eTIV 0.191. These come from a regularised model and are not accompanied by confidence intervals or p-values.

## Limitations

- The test set is small (47 subjects). The three models differ by at most one correctly classified subject, and the bootstrap intervals overlap, so no model is claimed to be better than another.
- Bootstrap intervals do not include training variability, and cross-validation uses the test subjects too.
- MMSE dominates the importance in both tree models. It is a cognitive measure closely related to what CDR rates, so a sensitivity analysis without MMSE would be informative.
- SES is missing far more often in the CDR > 0 group (17.00%) than in the CDR = 0 group (1.48%), so missingness is related to the outcome. Median imputation without a missingness indicator does not account for this.
- The modelled sample (235 subjects with a CDR) is mainly older adults and excludes the younger participants, so results do not describe the whole OASIS-1 population. Age differs between the groups (about 7.7 years).
- eTIV and ASF carry overlapping information, and Age and nWBV are related, which affects how coefficients and importances can be interpreted.
- Group comparisons were run on seven variables without a multiplicity correction.
- Single dataset with no external validation. Feature importance is not causal.

## How to run

The notebook was written for Google Colab: the first cell opens an upload dialog (`google.colab.files.upload()`), and the XGBoost cell runs `!pip -q install xgboost`.

- **In Colab:** upload the OASIS-1 `.xlsx` file when prompted and run the cells in order.
- **Locally:** install the requirements, put the data file in `data/`, and replace the first two cells with:

```python
import pandas as pd
df = pd.read_excel("data/<your-oasis-file>.xlsx")
```

```bash
git clone https://github.com/yadavanshudstcims2025-cell/oasis-cdr-cognitive-impairment-classification.git
cd oasis-cdr-cognitive-impairment-classification
pip install -r requirements.txt
jupyter notebook notebooks/oasis_cdr_classification.ipynb
```

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/            # place the OASIS file here (not tracked)
└── notebooks/
    └── oasis_cdr_classification.ipynb
```

## References

- Marcus, D. S., Wang, T. H., Parker, J., et al. (2007). Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. *Journal of Cognitive Neuroscience*, 19(9), 1498-1507.
- Morris, J. C. (1993). The Clinical Dementia Rating (CDR): Current version and scoring rules. *Neurology*, 43(11), 2412-2414.

## Author

Anshu Yadav, M.Sc. Statistics and Computing, DST CIMS, Banaras Hindu University.
