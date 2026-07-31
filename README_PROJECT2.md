# Project 2: Explainability

This Django application studies interpretable and explainable machine learning
using the Palmer Penguins dataset. The current implementation covers Tasks 1-5
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

Tasks 1-5 are implemented.

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

The selected model follows the PDF objective and maximizes:

```text
test accuracy - lambda * normalized complexity
```

For decision trees, complexity is the number of leaves. In the interface, this
leaf count is normalized by the largest candidate-tree size so the accuracy and
complexity terms have comparable scales. A low lambda focuses on accuracy; a
high lambda gives more importance to a smaller tree.

Task 2 extends Task 1 in the same interface. The accuracy, leaf count, and tree
shown for Task 1 therefore correspond to the model selected by Task 2.

### Task 3: Logistic Regression Complexity

- Trains L1-regularized logistic regression models with different `C` values.
- Uses the number of non-zero coefficients as the model complexity measure.
- Lets the same lambda slider select the best accuracy-complexity trade-off.
- Uses `l1_ratio=1.0` with the SAGA solver to avoid deprecated scikit-learn
  `penalty="l1"` warnings.

### Task 4: Counterfactual Explanations

- Lets the user select a penguin example and a desired species.
- Generates local random variations around the selected example.
- Handles numeric features with Gaussian noise and categorical features by
  sampling valid categories.
- Ranks matching counterfactuals by MAD-weighted L1 distance.
- Retries with larger samples and wider variance if no counterfactuals are
  found on the first attempt.

### Task 5: Feature Effect Plots

- Lets the user choose a numeric feature.
- Computes PDP and ALE values in project code rather than with a dedicated
  explainability library.
- Displays one probability curve per species.
- Links the PDP and ALE plots to the currently selected model type and lambda.

## HCAI Concepts Applied

These design choices follow concepts from the explainability and
interpretability lectures:

- **Interpretable models:** decision trees are understandable by design rather
  than explained only after training.
- **Accuracy-interpretability trade-off:** predictive performance is shown
  together with model complexity.
- **Complexity regularization:** the number of leaves is used as the
  decision-tree complexity measure; the number of non-zero logistic-regression
  coefficients is used as the linear-model complexity measure.
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
    |-- tests.py                      # Automated Project 2 tests
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

The tests cover dataset cleaning, readable feature labels, decision-tree
metrics, lambda validation, candidate selection, counterfactual retry behavior,
page rendering, and tree-image generation.
