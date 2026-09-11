# Project 4: Preference Elicitation

Project 4 designs a user study for comparing two movie-preference elicitation
interfaces: repeated pairwise choices between two movies and ranking ten movies
from most to least preferred.

Project page: <http://127.0.0.1:8000/project4/>

The assignment does not require running the study with real participants. The
implementation provides a participant-ready prototype, a downloadable PDF
report, and a session-based recommendation result after the participant submits
both tasks.

## Assignment Tasks

1. Build a feature representation for the IMDB 5000 Movie Dataset.
2. Propose a ranking-model extension of Bradley-Terry.
3. Design a user study comparing pairwise choice and ranking elicitation.
4. Implement a user interface for preference elicitation.

Tasks 1-4 are implemented.

## Implemented Functionality

### Feature Representation

- `feature_representation.py` cleans movie metadata and builds a numeric
  content-based feature representation.
- Movie features include genres, keywords, director information, and other
  metadata-derived signals.
- The interface explains how many movies and model features are available.

### Pairwise and Ranking Elicitation

- The participant completes five pairwise movie choices.
- The participant ranks ten movies from most preferred to least preferred.
- Ranking works with drag-and-drop and Up/Down buttons so the interaction does
  not depend on one input style.
- Movies already shown during elicitation are excluded from final
  recommendations.

### User Study Design and Report

- The PDF report explains the Bradley-Terry motivation and the Plackett-Luce
  ranking extension.
- The report describes hypotheses, within-subject design, counterbalancing,
  recruitment, six-step procedure, piloting, exclusion criteria, planned tests,
  metrics, and privacy/GDPR handling.
- The landing page keeps the PDF report accessible while making **Start Study**
  the primary participant action.

### Recommendations

- Submitted choices are stored in the Django session.
- The backend estimates a simple preference vector and returns recommended
  movies.
- Recommendation labels show short feature-based reasons behind the score.
- Users can reset the browser session and sample a fresh study set.

## HCAI Concepts Applied

- **Preference elicitation:** the system learns user preferences from limited
  pairwise and ranking interactions.
- **Human control:** the participant actively chooses and ranks movies, then can
  reset the session.
- **Transparency:** the page explains the task flow, session-only storage, and
  recommendation basis.
- **User-centered interaction:** ranking supports both drag-and-drop and button
  controls.
- **Study readiness:** the interface, report, timing capture, feedback fields,
  and reset flow make the prototype usable as a study material package.
- **Appropriate trust:** recommendations include explanation labels rather than
  only showing scores.

## Directory Structure

```text
project4/
|-- feature_representation.py        # Movie cleaning and feature extraction
|-- preference_learning.py           # Preference fitting and recommendation logic
|-- views.py                         # Landing page, study page, APIs, report
|-- tests.py                         # Automated Project 4 tests
|-- urls.py
|-- movie_metadata.csv               # Movie dataset used by the prototype
|-- templates/project4/index.html    # Landing and participant study interface
|-- templates/project4/report_pdf.html
`-- static/project4/style.css
```

## Run the Project

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python manage.py runserver
```

Open <http://127.0.0.1:8000/project4/>.

## Check and Test

```powershell
python manage.py check
python manage.py test project4 --verbosity 1
```

## Suggested Evaluation Path

1. Open the Project 4 page.
2. Download the PDF report to inspect the study design and modeling choices.
3. Use **Start Study** to open the participant interface.
4. Complete the five pairwise choices.
5. Reorder the ten ranking movies by dragging or using the Up and Down buttons.
6. Add optional per-design feedback and submit the ranking task.
7. Review the recommended movies and explanation labels.
8. Use **Reset Study** to clear the session and sample a fresh set.
