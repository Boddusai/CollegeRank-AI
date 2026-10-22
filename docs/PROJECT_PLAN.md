# CollegeRank AI - Implementation Plan

## Phase 1: JEE Main / JoSAA
1. Obtain verified historical data.
2. Normalize schema.
3. Clean and validate.
4. EDA.
5. Create temporal splits: 2021-2023 / 2024 / 2025.
6. Train Logistic Regression, Random Forest and XGBoost.
7. Compare validation metrics.
8. Refit selected model on train + validation.
9. Evaluate on 2025.
10. Save model and metrics.
11. Build API.
12. Build React UI.

## Phase 2
Reuse the architecture for the remaining nine exams, with exam-specific
data cleaners, schemas and model artifacts where required.
