# Project 2: Explainability

This Django application studies interpretable machine learning using the
Palmer Penguins dataset. The current implementation completes Tasks 1 and 2
from the Project 2 assignment.

Project page: <http://127.0.0.1:8000/project2/>

## Assignment Tasks

1. **Decision tree:** train and display a decision tree, its test accuracy, and
   its number of leaves.
2. **Model complexity:** train trees with different complexity limits and let
   the user control the accuracy-simplicity trade-off using lambda.
3. **Logistic regression:** repeat the complexity comparison with logistic
   regression.
4. **Counterfactuals:** generate local counterfactual explanations.
5. **Feature effects:** implement PDP and ALE plots for numeric features.

Tasks 1 and 2 are implemented. Tasks 3-5 remain future work.

## Implemented Functionality

### Task 1: Interpretable Decision Tree

- Loads the Palmer Penguins dataset.
- Predicts `species`: Adelie, Chinstrap, or Gentoo.
- Uses seven numeric and categorical input features.
- Removes rows missing a required feature or target: 333 of 344 rows remain.
- One-hot encodes `island` and `sex`.
- Uses a reproducible stratified 80/20 train/test split.
- Displays test accuracy and the actual number of leaves.
- Renders the selected decision tree with readable feature names.
- Shows a data-transparency panel explaining features, classes,
  preprocessing, and missing-value handling.

### Task 2: Accuracy-Complexity Trade-off

- Trains trees with `max_leaf_nodes` values:
  `2, 3, 4, 5, 6, 8, 10, 12, 15`.
- Uses the same train/test split for every candidate.
- Provides a lambda slider from `0.00` to `1.00`.
- Displays every candidate's accuracy, actual leaves, and selection score.
- Highlights and renders the selected tree.
- Explains that lambda is a user preference, while `max_leaf_nodes` is the
  tree-training parameter.

The selected model minimizes:

```text
(1 - test accuracy) + lambda * normalized leaf count
```

Test error (1 - test accuracy) is used because the objective is minimized. Leaf count is normalized
so that accuracy and complexity have comparable scales. A low lambda focuses
on accuracy; a high lambda gives more importance to a smaller tree.

Task 2 extends Task 1 in the same interface. The accuracy, leaf count, and tree
shown for Task 1 therefore correspond to the model selected by Task 2.

## HCAI Concepts Applied

These design choices follow concepts from the explainability and
interpretability lectures:

- **Interpretable models:** decision trees are understandable by design rather
  than explained only after training.
- **Accuracy-interpretability trade-off:** predictive performance is shown
  together with model complexity.
- **Complexity regularization:** the number of leaves is used as the
  complexity measure.
- **Human control:** the lambda slider lets the user express a preference
  between accuracy and simplicity.
- **Recipient-aware explanation:** technical values are accompanied by
  plain-language guidance.
- **Transparency:** the interface exposes data preparation, model candidates,
  accuracy, complexity, and the selection score.
- **Appropriate trust:** explanations support inspection but do not prove that
  the model is correct, fair, or causally valid.
- **Reproducibility:** all candidate models use the same fixed, stratified
  train/test split.

The displayed tree represents predictive associations in this dataset. Its
branches should not be interpreted as biological causes.

## Directory Structure

```text
HCAI-PBL/
|-- manage.py
|-- requirements.txt
|-- pbl/                              # Django settings and main URLs
|-- home/                             # Project navigation
`-- project2/
    |-- README.md
    |-- views.py                      # Data, training and model selection
    |-- tests.py                      # Automated Task 1 and Task 2 tests
    |-- urls.py
    |-- templates/project2/index.html # Project 2 interface
    `-- static/project2/style.css     # Project-specific styling
```

Generated tree images are written to `media/project2/` at runtime.

## Run the Project

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python manage.py runserver
```

Open <http://127.0.0.1:8000/project2/>.

## Check and Test

```powershell
python manage.py check
python manage.py test project2 --verbosity 2
```

The current seven tests cover dataset cleaning, readable feature labels,
decision-tree metrics, lambda validation, candidate selection, page rendering,
and tree-image generation.
