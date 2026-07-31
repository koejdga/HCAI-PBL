# Project 2: Explainability

This Django app implements the Palmer Penguins explainability project.

## Implemented Tasks

- **Task 1:** decision tree, rendered tree image, test accuracy, and leaf count.
- **Task 2:** lambda-based accuracy-complexity selection for decision trees.
- **Task 3:** logistic regression with L1 sparsity and non-zero coefficients as
  the complexity measure.
- **Task 4:** local counterfactual search with MAD-weighted L1 ranking and
  retry attempts when no counterfactual is found initially.
- **Task 5:** manually implemented PDP and ALE plots with one curve per species.

## Model Selection Objective

The interface follows the assignment objective:

```text
test accuracy - lambda * normalized complexity
```

The selected model is the candidate with the highest score. Complexity is
normalized so the accuracy and complexity terms are on comparable scales.

## Run

```powershell
cd "D:\TUHH\Data Sci Curriculum\Courses Notes\SoSe2026\Human-Centric AI\PBL"
.\.venv\Scripts\Activate.ps1
python manage.py runserver
```

Open <http://127.0.0.1:8000/project2/>.

## Test

```powershell
python manage.py test project2 --verbosity 2
```
