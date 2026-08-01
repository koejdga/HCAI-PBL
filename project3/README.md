# Project 3: Active Learning for Learning-to-Defer

Project 3 studies human-AI collaboration for AG News classification. The system
trains a baseline classifier, simulates experts, compares learning-to-defer
policies, uses active learning to estimate expert competence, and generates a
PDF report from the interface.

Project page: <http://127.0.0.1:8000/project3/>

## Assignment Tasks

1. Train a baseline classifier on AG News and report test accuracy.
2. Implement simulated experts and analyze their strengths and weaknesses.
3. Implement learning-to-defer strategies and report deferral quality.
4. Use active learning to query experts efficiently and estimate competence.
5. Optional: provide a human-expert interaction interface.

Tasks 1-5 are implemented. The project also includes a bonus stream-based
selective sampling strategy.

## Implemented Functionality

### Baseline Classifier

- Uses AG News text data.
- Trains a TF-IDF + linear classifier baseline.
- Reports test accuracy and per-class performance.

### Simulated Experts

- Supports realistic and trivial expert profiles.
- Allows one or two simulated experts.
- Lets users configure expert domain competence and query cost.
- Shows expert accuracy overall and by class.
- Uses a default realistic expert if no expert is selected.

### Learning to Defer

- Compares baseline-only, expert-only, confidence-threshold deferral,
  competence-aware deferral, and learning-to-defer models.
- Reports accuracy, deferral rate, useful deferrals, harmful deferrals, total
  cost, and net benefit.
- Shows query allocation by class and expert.

### Active Learning

- Implements balanced uncertainty, uncertainty-only, random sampling, and
  stream selective sampling.
- Displays queried examples and estimated expert competence by class.
- Shows strategy comparison metrics such as agreement, competence, retrained
  model accuracy, redundancy, and diversity.
- Provides a query-budget recommendation using a convergence table and plot.

### Human Expert Interface

- Lets the user act as the expert for selected active-learning queries.
- Tracks answered labels, correctness, and human-expert accuracy.

### PDF Report

- The interface includes a PDF report section.
- Users can save expert configurations and download a report containing the
  experiment setup, design choices, tables, plots, and results.

## HCAI Concepts Applied

- **Learning to defer:** the model can decide whether to predict or ask an
  expert.
- **Human-AI collaboration:** expert reliability, cost, and competence are part
  of the decision process.
- **Active learning:** expert queries are selected to learn competence
  efficiently.
- **Human effort awareness:** query budgets and convergence results show how
  much expert effort is being requested.
- **Appropriate trust:** the interface separates baseline, expert-only, and
  human-AI team performance.
- **Interactive ML:** users can configure experts and optionally provide labels
  themselves.

## Runtime Notes

Project 3 performs text classification, learning-to-defer training, active
learning analysis, plot generation, and PDF generation. These computations are
heavier than Projects 1 and 2.

Recommended quick demo settings:

```text
Training examples: 1000-2000
Test examples: 500-1000
Deferral rate: 0.3
Expert query budget: 20-40
```

Larger values can be used, but they will take longer. The page shows loading
feedback and falls back to a normal reload if an AJAX update takes too long.

## Directory Structure

```text
project3/
|-- views.py                         # Page, report, result assembly
|-- core/
|   |-- active_learning.py           # Query strategies and human expert flow
|   |-- deferral.py                  # Baseline, experts, deferral policies
|   |-- learning_to_defer.py         # L2D linear and neural models
|   `-- utils.py                     # Dataset loading and plot helpers
|-- tests.py                         # Automated Project 3 tests
|-- urls.py
|-- templates/project3/index.html    # Main Project 3 interface
|-- templates/project3/report_pdf.html
`-- static/project3/style.css
```

Generated plots and report assets are written to `media/project3/` at runtime.

## Run the Project

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python manage.py runserver
```

Open <http://127.0.0.1:8000/project3/>.

## Check and Test

```powershell
python manage.py check
python manage.py test project3 --verbosity 1
```

For a faster smoke test during development:

```powershell
python manage.py test project3.tests.Project3ExperimentTests.test_stream_selective_sampling_respects_query_budget --verbosity 1
```

## Suggested Evaluation Path

1. Open the Project 3 page.
2. Use the recommended quick demo settings above.
3. Review baseline classifier results.
4. Configure a realistic expert and apply expert settings.
5. Compare policy results and query allocation.
6. Review active-learning strategy comparison.
7. Try stream selective sampling in the active-learning strategy radio buttons.
8. Save a configuration and download the PDF report.
