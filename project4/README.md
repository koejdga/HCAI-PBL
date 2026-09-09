# Project 4: Preference Elicitation

Project 4 designs a user study for comparing two movie-preference elicitation
interfaces:

1. repeated pairwise choices between two movies;
2. ranking ten movies from most to least preferred.

The assignment does not require running the study. The implementation provides
a participant-ready prototype, a downloadable PDF report, and a session-based
recommendation result after the participant submits both tasks.

## Quick Evaluation Path

1. Start the Django server from the repository root.
2. Open <http://127.0.0.1:8000/project4/>.
3. Use **Start Study** to open the participant interface.
4. Complete the five pairwise choices.
5. Reorder the ten ranking movies by dragging or using the Up and Down buttons.
6. Add optional per-design feedback and submit the ranking task.
7. Review the recommended movies and explanation labels.
8. If the backend provides separate model outputs, compare pairwise-only,
   ranking-only, and combined recommendation groups.
9. Download the PDF report from the final report section, or from the landing page.
10. Use **Reset Study** to clear the session and sample a fresh set.

## Assignment Coverage

- **Task 1:** `feature_representation.py` builds a numeric content-based movie
  representation from the IMDB 5000 Movie Dataset.
- **Task 2:** the report proposes Plackett-Luce as the ranking extension of
  Bradley-Terry. The prototype implements a lean pairwise-decomposition model
  for ranking observations.
- **Task 3:** the report defines the user-study hypothesis, within-subject
  design, counterbalancing, recruitment plan, six-step protocol, pilot plan,
  exclusion criteria, planned statistical tests, and privacy/GDPR handling.
- **Task 4:** the Django UI provides the landing page, study interface, PDF
  download, session reset, backend submission endpoints, per-task timing,
  separated feedback, accessible ranking controls, and recommendations.

## Human-Centric Design Choices

- The landing page makes the participant study the primary action while keeping
  the PDF report accessible for evaluation.
- The participant sees that choices are stored only for the browser session.
- Per-task timing and separate effort/clarity feedback make the comparison
  more usable as a real user-study prototype.
- Ranking works with drag-and-drop and buttons so the interface does not assume
  one input style works for everyone.
- Movies already used during elicitation are excluded from recommendations.
- Recommendation labels expose the feature signals behind the score.
- The model remains intentionally simple so the preference vector can be
  explained in terms of movie metadata.

## Main Files

- `views.py`: landing page, study page, JSON endpoints, session handling.
- `feature_representation.py`: movie cleaning, feature extraction, labels.
- `preference_learning.py`: comparison conversion, preference fitting,
  recommendation scoring, explanation contributions.
- `templates/project4/index.html`: landing and participant interface.
- `templates/project4/report_pdf.html`: PDF report for the Project 4 tasks.
- `static/project4/style.css`: Project 4 styling aligned with the shared TUHH
  visual system.

## Test

```powershell
python manage.py test project4 --verbosity 1
```
