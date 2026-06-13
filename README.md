# Human-Centric Artificial Intelligence PBL

This repository contains the project-based learning work for the TUHH
Human-Centric Artificial Intelligence course. The projects share one Django
website with a central homepage and a separate Django app for each project.

## Project 1 at a Glance

Project 1 is a browser-based supervised-learning assistant. A user uploads a
CSV file, reviews how the system interpreted the data, creates visualizations,
chooses a model, and compares the trained model with a simple baseline.

**Five-minute evaluation path for the professor:**

1. Follow the installation commands below.
2. Open <http://127.0.0.1:8000/project1/>.
3. Upload a CSV dataset such as Iris, with the target in the final column.
4. Review **Dataset Setup** and **Dataset & Assumptions**.
5. Create one feature plot and one scatter plot.
6. Under **Model Training**, select a model and choose **Train model**.
7. Compare the three result cards: trained model, simple baseline, and
   observed difference.

The key human-centric feature is not automatic training alone. The interface
keeps important assumptions visible and lets the user review or change them
before interpreting the result.

## Group Members

| Name | Matriculation number |
| --- | --- |
| Sofiia Budilova | 675972 |
| Ashutosh Chatterjee | 672405 |
| Gauri Gajanan Amin | 670328 |

## Repository Structure

```text
HCAI-PBL/
|-- home/              # Homepage, group members, and project navigation
|-- project1/          # Project 1 supervised-learning application
|-- demos/             # Course examples for uploads and plots
|-- pbl/               # Django project settings and root URL configuration
|-- static/            # Shared CSS and images
|-- templates/         # Shared base templates
|-- media/             # Generated visualization images
|-- manage.py          # Django command-line entry point
`-- requirements.txt   # Python dependencies
```

Project 1 is implemented in `project1/`. Its main backend logic is in
`project1/views.py`, and its interface is in
`project1/templates/project1/upload.html`.

## Requirements

- Python 3
- A current web browser
- The packages listed in `requirements.txt`

## Installation

Clone the repository and enter its root directory:

```powershell
git clone https://github.com/koejdga/HCAI-PBL.git
cd HCAI-PBL
```

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create the Django database tables:

```powershell
python manage.py migrate
```

The migration step is required because Django sessions are used to retain the
uploaded dataset and generated results. Without it, uploads can fail with
`no such table: django_session`.

Start the development server:

```powershell
python manage.py runserver
```

Open:

- Homepage: <http://127.0.0.1:8000/>
- Project 1: <http://127.0.0.1:8000/project1/>

Stop the server with `Ctrl+C`.

## Project 1: Supervised Learning Interface

Project 1 implements the four tasks from the assignment:

1. Display the group members on the homepage using the Python view.
2. Provide a dedicated Django app linked from the homepage.
3. Upload, inspect, and visualize a CSV dataset.
4. Train and evaluate supervised-learning models for multiple
   hyperparameter values.

The application supports both classification and regression. It is intended as
an educational interface for exploring the decisions involved in a basic
machine-learning pipeline.

## CSV Format

The uploaded file must:

- use CSV format;
- contain a header row with unique, non-empty column names;
- contain at least one feature column and one target column;
- contain at least one data row;
- use the same number of values in every row.

By default, the last column is selected as the target, as specified in the
assignment. The user can select a different target in the interface.

The following values are treated as missing:

```text
empty value, NA, N/A, NaN, null, none, ?
```

Columns are detected as numeric when every non-missing value can be converted
to a number. Other columns are treated as categorical.

## Using Project 1

The page follows the same order as a basic ML workflow:

```text
Upload data -> Review setup -> Inspect plots -> Train -> Compare with baseline
```

1. Open the Project 1 page.
2. Select a CSV file and choose **Visualize**.
3. Review the detected target, task type, selected features, missing values,
   assumptions, and data-quality warnings.
4. Optionally change:
   - the target column;
   - classification or regression;
   - included feature columns;
   - whether incomplete rows are retained or removed.
5. Inspect the automatically generated dataset overview.
6. Generate feature-versus-target plots.
7. Generate scatter plots for two selected features.
8. Choose a model and a test split.
9. Select **Train model** to compare model settings and the naive baseline.
10. Use **Clear dataset** to remove the current dataset and results from the
    browser session.

Changing the dataset configuration clears existing plots and model results so
that stale results are not presented for a different setup.

## Data Preparation

The application performs the following preprocessing during training:

- numeric missing feature values: median imputation;
- categorical missing feature values: most-frequent-value imputation;
- numeric features: standard scaling;
- categorical features: one-hot encoding;
- unseen categorical values: ignored by the encoder.

The user may instead remove every row containing a missing value. Missing
target values are not accepted for training.

Columns that resemble identifiers, such as `id`, `row_id`, `index`, or a
sequential unique numeric column, are excluded by default. The user can include
them again when they contain meaningful information.

## Task Detection

A categorical target is treated as classification. A numeric target is also
treated as classification when it contains a small number of integer-coded
values; otherwise, it is treated as regression.

This automatic detection is a heuristic. The inferred task is shown to the
user and can be overridden. Regression requires a numeric target.

## Models and Evaluation

### Classification

- Logistic Regression:
  - `C = 0.1`
  - `C = 1`
  - `C = 10`
- K-Nearest Neighbors:
  - `k = 3`
  - `k = 5`
  - `k = 7`, when the training set is large enough
- Metric: accuracy

For Logistic Regression, smaller `C` values apply stronger regularization and
larger values apply weaker regularization.

### Regression

- Ridge Regression:
  - `alpha = 0.1`
  - `alpha = 1`
  - `alpha = 10`
- K-Nearest Neighbors:
  - `k = 3`
  - `k = 5`
  - `k = 7`, when the training set is large enough
- Metrics:
  - R2 score
  - root mean squared error (RMSE)

The user can select a 20%, 30%, or 40% held-out test split. The split uses a
fixed random seed of `42` for reproducibility. Classification uses a stratified
split when possible.

## Baseline Comparison

A score is difficult to judge without a reference. For example, 70% accuracy
may appear strong, but not if 80% of the data belongs to one class. Project 1
therefore compares each trained model against a deliberately simple baseline
evaluated on exactly the same train/test split:

- classification: always predict the most frequent training class;
- regression: always predict the mean training target.

The results section shows:

- the best trained-model score;
- the baseline score;
- the observed difference;
- whether the model performed better than, similarly to, or worse than the
  baseline;
- the baseline RMSE for regression;
- the number of training and test rows.

The comparison is presented as:

```text
Trained model | Simple baseline | Observed difference
```

For classification, scores are displayed as percentages. For regression, R2
and RMSE are accompanied by short explanations. Beating the baseline indicates
that the model found predictive signal beyond a naive rule. It does not
establish that the model is reliable, fair, or suitable for real-world
deployment.

## Visualizations

The interface supports:

- target-distribution summaries;
- feature-distribution overviews;
- correlation views for numeric data;
- feature-versus-target plots;
- scatter plots between selected features.

Matplotlib uses a non-interactive backend and saves generated figures under
`media/` so Django can display them. Dynamically requested plots can be added
and removed without reloading the entire page.

## Dataset Assumptions and Warnings

Before training, the interface communicates:

- row and feature counts;
- selected and default target columns;
- detected and selected task type;
- missing-value policy;
- test-split size;
- preprocessing behavior.

Warnings may be shown for:

- missing targets;
- missing values;
- rows removed because of missing values;
- likely identifier columns;
- class imbalance;
- high-cardinality categorical features;
- datasets containing fewer than 30 rows.

These notices are intended to help users review the data and system assumptions
before interpreting model scores.

## Human-Centric AI Principles

The interface applies ideas introduced in the course lectures:

### Human control over the ML pipeline

The introductory lecture emphasizes that humans influence data collection,
algorithm selection, hyperparameters, evaluation criteria, and intended use.
The interface keeps the user involved by allowing control over the target,
features, task type, cleaning policy, model, and test split.

### Transparency of assumptions

The application explicitly shows what it inferred and what preprocessing it
will perform. This makes hidden technical choices visible and reviewable.

### Support for appropriate trust

The explainability lectures distinguish blindly trusting a model from having
enough information to decide whether trust is warranted. Baseline comparison,
quality warnings, preprocessing notes, and neutral result messages help users
judge whether a score represents meaningful improvement.

### Understandability for non-experts

The course notes that many users are not machine-learning experts and may not
understand models or hyperparameters. The interface therefore provides
plain-language explanations for task detection, baseline behavior, data
cleaning, model settings, and evaluation metrics.

### Data awareness

Human-created and selected data can contain missing values, sampling problems,
biases, leakage, and unsuitable proxy features. The application cannot remove
all such problems automatically, but it surfaces possible identifier columns,
class imbalance, missing data, small samples, and high-cardinality features for
human review.

### Interpretability and visual inspection

The application uses comparatively understandable model families and provides
visualizations that allow users to inspect feature relationships and target
structure before training.

### Contextual evaluation

The lectures emphasize that predictive performance is not the only relevant
criterion in real applications. The interface avoids declaring a model
"good" solely from its score and warns that beating a baseline does not imply
deployment readiness.

### User-centered evaluation

The user-studies lecture highlights usability, effectiveness, trust, mental
models, representative users, and piloting. The current interface supports
these goals through visible assumptions and guided interaction, but a formal
user study has not yet been conducted.

## Testing

Run Django's system checks:

```powershell
python manage.py check
```

Run the Project 1 automated tests:

```powershell
python manage.py test project1
```

The current tests verify:

- the classification majority-class baseline;
- the regression mean baseline, R2, and RMSE values;
- better, similar, and worse baseline-comparison messages.

Recommended manual checks include:

- a balanced classification dataset such as Iris;
- a numeric regression dataset;
- missing feature values;
- missing target values;
- a likely ID column;
- imbalanced target classes;
- small datasets;
- each model and test-split option;
- desktop and narrow browser layouts.

## Current Limitations

- Hyperparameter settings are currently compared on the held-out test set.
  A future version should select hyperparameters using cross-validation on the
  training data and evaluate the selected model once on the untouched test set.
- Automatic task detection is heuristic and may require user correction.
- Accuracy can conceal poor performance on minority classes.
- The application does not currently report precision, recall, F1 score,
  confusion matrices, confidence intervals, or prediction uncertainty.
- No fairness or subgroup analysis is implemented.
- Data-quality warnings cannot detect every source of bias, leakage, or
  unrepresentative sampling.
- Uploaded dataset contents and results are stored in the Django session for
  the current browser session.
- Generated plot files are stored locally in `media/`.

## Assignment Reference

Project 1 corresponds to the course assignment **Project 1: Automated Machine
Learning**. The assignment requires one shared Django repository, a homepage
with group information, a Project 1 app, CSV upload and visualization, and an
end-to-end supervised-learning pipeline with model selection, data splitting,
multiple hyperparameter values, and evaluation.
