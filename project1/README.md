# Project 1: Automated Machine Learning

Project 1 is a supervised-learning interface for uploading a CSV dataset,
inspecting assumptions, visualizing features, training models, and comparing
the trained model with a simple baseline.

Project page: <http://127.0.0.1:8000/project1/>

## Assignment Tasks

1. Create and connect the Project 1 Django app.
2. Add the project link from the homepage.
3. Upload a CSV file and visualize the dataset.
4. Train supervised-learning models with train/test splitting, several
   hyperparameter values, and evaluation scores.

Tasks 1-4 are implemented.

## Implemented Functionality

## How to Use the Interface

Project 1 is organized as a step-by-step page:

1. **Data Management** lets the user either choose a built-in example dataset
   or upload a custom CSV file. Example datasets are stored in
   `project1/example_datasets/` and can run the full workflow without upload.
   Uploaded files appear separately under **Uploaded Datasets**.
2. **Dataset Audit** appears after a dataset is selected. It shows column
   statistics, missing-value information, correlation information, feature
   plots, and scatter/distribution plot tools.
3. **Model Training** appears after dataset setup. It lets the user select a
   model, adjust available hyperparameters, train the model, compare against a
   baseline, and run the "Train All and Compare" workflow.
4. The right-side **Contents** menu provides direct navigation between the main
   sections and subsections.

### CSV Upload and Dataset Review

- Accepts CSV files with a header row.
- Uses the last column as the default target, as described in the assignment.
- Lets the user change the target, task type, included features, and missing
  value handling.
- Rejects non-CSV files with a clear upload message.
- Detects likely ID columns and excludes them by default.

### Visualization

- Shows dataset summary information and data-quality warnings.
- Generates feature-versus-target plots.
- Generates scatter plots for two selected features.
- Generates univariate feature distributions and a numeric correlation matrix.
- Prevents selecting the same feature for both scatter-plot axes.
- Lets users click plots to view them larger.

### Model Training

- Supports classification and regression workflows.
- Uses a held-out test split selected by the user.
- Compares several model settings:
  - Logistic Regression, K-Nearest Neighbors, Support Vector Machines,
    Random Forest, and Gaussian Naive Bayes for classification.
  - Linear Regression, Ridge Regression, Lasso Regression,
    K-Nearest Neighbors, Support Vector Machines, and Random Forest for
    regression.
- Shows the trained model result next to a simple baseline:
  - majority-class baseline for classification;
  - mean-target baseline for regression.
- Includes a "Train All and Compare" view plus a PDF summary report for the
  model comparison.

## HCAI Concepts Applied

- **Human control:** users can inspect and adjust target, task type, selected
  features, and missing-value handling.
- **Transparency:** assumptions, data-quality warnings, preprocessing, and
  model choices are visible before interpreting results.
- **Appropriate trust:** model performance is compared against a simple
  baseline instead of being shown alone.
- **Understandable feedback:** metric explanations and optional info popups use
  plain language.
- **User-centered interaction:** the page follows a clear workflow from upload
  to review, visualization, training, and comparison.
- **Bias awareness:** the interface shows data-quality warnings and reminds the
  user that cleaning, visualization, and model comparison do not automatically
  remove dataset bias or representation bias.

## Directory Structure

```text
project1/
|-- core/
|   |-- dataset.py                 # CSV parsing, dataset configuration, audits
|   |-- training.py                # Model training, baselines, comparison logic
|   `-- visualization.py           # Matplotlib plot generation
|-- example_datasets/              # Built-in demo datasets
|-- static/project1/style.css      # Project-specific styling
|-- templates/project1/index.html  # Main Project 1 page
|-- templates/project1/components/ # Reusable UI sections
|-- tests.py                       # Automated Project 1 tests
|-- urls.py
`-- views.py                       # Django views and session workflow
```

Generated plot images are written to `media/project1/` at runtime.

## Run the Project

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python manage.py runserver
```

Open <http://127.0.0.1:8000/project1/>.

## Check and Test

```powershell
python manage.py check
python manage.py test project1 --verbosity 1
```

## Suggested Evaluation Path

1. Open the Project 1 page.
2. Upload an Iris-style CSV file.
3. Review the dataset setup and assumptions.
4. Generate one feature plot and one scatter plot.
5. Train a classification model.
6. Compare the trained model result with the simple baseline.
