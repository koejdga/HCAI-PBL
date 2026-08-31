# Human-Centric Artificial Intelligence PBL

This repository contains the TUHH Human-Centric Artificial Intelligence
project-based learning work. The implementation is one Django website with a
central homepage and separate apps for Projects 1, 2, and 3.

## Group Members

| Name | Matriculation number |
| --- | --- |
| Sofiia Budilova | 675972 |
| Ashutosh Chatterjee | 672405 |
| Gauri Gajanan Amin | 670328 |

## Projects

| Project | Topic | App URL | Project README |
| --- | --- | --- | --- |
| Project 1 | Automated Machine Learning | `/project1/` | `project1/README.md` |
| Project 2 | Explainability | `/project2/` | `project2/README.md` |
| Project 3 | Active Learning for Learning-to-Defer | `/project3/` | `project3/README.md` |

## Repository Structure

```text
HCAI-PBL/
|-- home/              # Homepage, group members, and project navigation
|-- project1/          # Automated ML interface
|-- project2/          # Explainability interface
|-- project3/          # Active learning and learning-to-defer interface
|-- pbl/               # Django settings and root URL configuration
|-- static/            # Shared CSS
|-- templates/         # Shared base templates
|-- media/             # Generated plots and reports at runtime
|-- manage.py
`-- requirements.txt
```

## Installation

Clone the repository and enter the repository root:

```powershell
git clone https://github.com/koejdga/HCAI-PBL.git
cd HCAI-PBL
```

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create Django database tables:

```powershell
python manage.py migrate
```

Start the development server:

```powershell
python manage.py runserver
```

Open:

- Homepage: <http://127.0.0.1:8000/>
- Project 1: <http://127.0.0.1:8000/project1/>
- Project 2: <http://127.0.0.1:8000/project2/>
- Project 3: <http://127.0.0.1:8000/project3/>

Stop the server with `Ctrl+C`.

## Project 1: Automated Machine Learning

Project 1 lets users upload a CSV dataset, review the detected data setup,
generate visualizations, train supervised-learning models, and compare the
trained model with a simple baseline.

Implemented assignment tasks:

1. Homepage group information from the Python view.
2. Project 1 Django app linked from the homepage.
3. CSV upload, dataset inspection, and visualization.
4. End-to-end supervised-learning pipeline with train/test splitting,
   hyperparameter comparison, and evaluation.

Special implementation aspects:

- Supports classification and regression.
- Lets users adjust target column, task type, selected features, and missing
  value handling.
- Detects likely ID columns.
- Shows baseline comparison for appropriate trust.
- Provides optional info popups and clickable larger plots.

See `project1/README.md` for details.

## Project 2: Explainability

Project 2 uses the Palmer Penguins dataset to demonstrate interpretable models,
complexity-aware model selection, counterfactual explanations, and global
feature-effect plots.

Implemented assignment tasks:

1. Decision tree with test accuracy and number of leaves.
2. Lambda slider for accuracy-complexity model selection.
3. Logistic regression with a suitable complexity measure.
4. Counterfactual explanation region linked to the selected model.
5. PDP and ALE plots implemented in project code.

Special implementation aspects:

- Uses normalized complexity in the lambda objective for clearer interaction.
- Uses readable feature names in tree visualizations.
- Includes a data-transparency panel.
- Explains that PDP and ALE summarize model behavior, not causal proof.

See `project2/README.md` for details.

## Project 3: Active Learning for Learning-to-Defer

Project 3 implements human-AI collaboration on AG News classification. It
trains a baseline classifier, simulates experts, compares learning-to-defer
policies, uses active learning to estimate expert competence, and generates a
PDF report.

Implemented assignment tasks:

1. Baseline AG News classifier with test accuracy.
2. Simulated experts with configurable competence and cost.
3. Learning-to-defer strategies with deferral-quality metrics.
4. Active learning strategies for expert competence discovery.
5. Optional human-expert interface.

Special implementation aspects:

- Supports one or two simulated experts.
- Includes confidence-threshold, competence-aware, linear L2D, and neural L2D
  policy comparisons.
- Adds a bonus stream selective sampling strategy.
- Shows query-budget convergence and recommended budget.
- Provides PDF report generation from saved configurations.
- Uses loading feedback and a fallback reload for long computations.

Default and recommended quick demo settings:

```text
Training examples: 2000
Test examples: 1000
Deferral rate: 0.3
Expert query budget: 20-40
```

See `project3/README.md` for details.

## HCAI Concepts Applied

- **Transparency:** assumptions, preprocessing, model choices, and selected
  features are visible to the user.
- **Human control:** users can adjust key decisions such as features, model
  type, lambda preference, expert settings, and query budget.
- **Interpretability:** Project 2 uses decision trees and complexity-aware
  selection.
- **Post-hoc explanation:** Project 2 includes counterfactuals, PDP, and ALE.
- **Appropriate trust:** Projects 1 and 3 compare model performance against
  baselines and expert/team alternatives.
- **Learning to defer:** Project 3 models when the system should predict and
  when it should defer to an expert.
- **Active learning:** Project 3 selects informative expert queries to estimate
  competence efficiently.
- **User-centered interaction:** the interface includes guided workflows,
  optional info popups, loading feedback, readable tables, and PDF reporting.

## Check and Test

Run Django checks:

```powershell
python manage.py check
```

Run project tests:

```powershell
python manage.py test project1 --verbosity 1
python manage.py test project2 --verbosity 1
python manage.py test project3 --verbosity 1
```

Project 3 tests and larger Project 3 runs can take longer because text
classification, learning-to-defer training, active-learning analysis, plot
generation, and PDF report generation are more computationally intensive than
Projects 1 and 2.

## Evaluation Checklist

1. Install dependencies and run `python manage.py check`.
2. Open the homepage and verify links to all projects.
3. Follow the suggested evaluation path in each project README.
4. Try Project 3 first with the recommended quick demo settings.
5. Open the app in Chrome and at least one other browser such as Edge or
   Firefox.
6. Resize the browser window and confirm that forms, tables, plots, and
   navigation remain usable.
