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
- Prevents selecting the same feature for both scatter-plot axes.
- Lets users click plots to view them larger.

### Model Training

- Supports classification and regression workflows.
- Uses a held-out test split selected by the user.
- Compares several model settings:
  - Logistic Regression and K-Nearest Neighbors for classification.
  - Ridge Regression and K-Nearest Neighbors for regression.
- Shows the trained model result next to a simple baseline:
  - majority-class baseline for classification;
  - mean-target baseline for regression.

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

## Directory Structure

```text
project1/
|-- forms.py                       # CSV upload validation
|-- views.py                       # Data parsing, plots, training, results
|-- tests.py                       # Automated Project 1 tests
|-- urls.py
`-- templates/project1/upload.html # Project 1 interface
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
