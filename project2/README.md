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

Tasks 1-5 are implemented, along with lecture-grounded explainability extensions from Lectures 2, 3, and 4.

## Implemented Functionality

### Task 1: Interpretable Decision Tree & Interactive Vector Visualization

- Loads the Palmer Penguins dataset with stratified 80/20 train/test split (333 cleaned rows).
- Renders an **interactive, resolution-independent SVG tree** directly in the DOM.
- **Node Hover Tooltips:** Hovering over any internal node or leaf dynamically reveals:
  - Gini impurity index.
  - Number of training samples reaching the node.
  - Interactive mini stacked bar chart of species class proportions (Adelie, Chinstrap, Gentoo).
- **Dedicated Compact Datapoint Path Explorer:**
  - A compact table (6 rows per page) separate from the counterfactuals section.
  - Selecting any penguin interactively illuminates its path from root to leaf with glowing green branches.
  - Renders a step-by-step decision sequence card displaying exact threshold comparisons and final prediction confidence.
- **CORELS-Style Rule List (Lecture 4):**
  - Extracts and formats all root-to-leaf decision paths into human-readable logical rules: `IF condition_1 AND condition_2 ... THEN PREDICT Species (confidence, sample support)`.
- Fallback high-resolution raster export (`decision_tree.png`) available for offline presentation.

### Task 2 & Rashomon Set Analysis (Lecture 4)

- Trains trees across complexity limits `max_leaf_nodes ∈ {2, 3, 4, 5, 6, 8, 10, 12, 15}`.
- User-controlled simplicity preference slider ($\lambda \in [0.00, 1.00]$):
  $$\text{Selection Score} = \text{test accuracy} - \lambda \times \frac{\text{leaves}}{\max(\text{leaves})}$$
- **The Rashomon Effect & Rashomon Ratio ($\mathcal{R}_{ratio}(\theta)$):**
  - Quantifies the fraction of candidate models whose test accuracy is within $\theta$ of the optimal model:
    $$\mathcal{R}_{ratio}(\theta) = \frac{|\mathcal{R}(\theta)|}{|\mathcal{F}|}$$
  - Renders a dedicated Rashomon banner and visual badges identifying all candidates in the Rashomon set.
  - Illustrates the lecture idea that several models can perform similarly, so a simpler interpretable model can often be chosen without giving up much accuracy.

### Task 3: Logistic Regression Interpretability (Lecture 2)

- L1-regularized Logistic Regression candidates across inverse regularization parameter $C \in [0.01, 10.0]$ using SAGA solver (`l1_ratio=1.0`).
- Complexity measured by the number of non-zero coefficients.
- **Weight Plot (Lecture 2, Slide 50):**
  - Generates horizontal bar charts of standardized feature coefficients $\beta_j$ across all three classes, providing a clear visual representation of feature importance and directional sign.
- **Odds Ratios & Log-Odds Table (Lecture 2, Slides 58–62):**
  - Formulates coefficients as log-odds and computes exact Odds Ratios $\text{OR} = e^{\beta_j}$.
  - Includes plain-language interpretation sentences (e.g., "A 1 mm increase in flipper length multiplies odds of Gentoo by 2.45x").

### Task 4: Actionable Counterfactuals & Actionability Constraints

- Lets the user pick an example penguin and a desired counterfactual target species.
- **Actionability Toggles (Actionability Constraints):**
  - Users can lock immutable features (Sex, Island, Year) so counterfactual explanations only suggest realistic, mutable changes.
- **Distance Metric with Categorical Penalty:**
  - Evaluates counterfactuals using Median Absolute Deviation (MAD)-weighted L1 distance for continuous features plus categorical mismatch penalty (1.0 per mismatched category):
    $$d(\mathbf{x}, \mathbf{x}') = \sum_{j \in \text{num}} \frac{|x_j - x_j'|}{\text{MAD}_j} + \sum_{k \in \text{cat}} \mathbb{I}[x_k \neq x_k']$$
- **Visual Diff Badges:**
  - Distinct green badges highlight precisely which features must change to achieve the alternative outcome.

### Task 5: The Feature Effect Trilogy (PDP, M-Plot, ALE)

- Computes three complementary global feature effect curves side-by-side for any chosen numeric feature:
  1. **Partial Dependence Plot (PDP):** Marginal effect assuming feature independence; susceptible to unlikely feature combinations.
  2. **Marginal Plot (M-Plot):** Averages predictions over the conditional distribution $P(X_C | X_S)$; addresses unlikely points but confounds correlated feature effects.
  3. **Accumulated Local Effects (ALE):** Accumulates local differences within neighborhood windows and centers around zero, isolating pure main effects.
- **Exact Partial Derivatives vs. Discretization (Task 5 Theoretical Question):**
  - For differentiable models (Logistic Regression), the local effect can theoretically be computed analytically via exact partial derivatives:
    $$\frac{\partial f(\mathbf{x})}{\partial x_S} = \beta_S \cdot f(\mathbf{x})(1 - f(\mathbf{x}))$$
  - For non-differentiable step functions (Decision Trees), partial derivatives are undefined ($\frac{\partial f}{\partial x} = 0$ almost everywhere with jump discontinuities at splits). Hence, finite differences / discretization into quantile bins are strictly necessary.

## HCAI Concepts Applied

These design choices follow concepts from the explainability and interpretability lectures:

- **Interpretable Models (Lecture 4):** Decision trees and sparse logistic models offer intrinsic glass-box interpretability without reliance on post-hoc surrogate approximations.
- **Rashomon Multiplicity (Lecture 4):** Acknowledging that multiple distinct models achieve near-optimal accuracy enables choosing simpler, safer models.
- **Log-Odds & Odds Ratios (Lecture 2):** Providing mathematically grounded, human-comprehensible descriptions of linear model weights.
- **Actionability & Plausibility (Lecture 3):** Ensuring counterfactual interventions respect real-world immutability constraints.
- **Feature Effect Distinctions (Lecture 3):** Explaining the mathematical trade-offs between PDP, M-Plot, and ALE.
- **Appropriate trust:** The interface notes that PDP, M-Plot, ALE, counterfactuals, and tree paths explain model behavior in this dataset; they should not be read as causal proof.

## Directory Structure

```text
HCAI-PBL/
|-- manage.py
|-- requirements.txt
|-- pbl/                              # Django settings and main URLs
|-- home/                             # Project navigation
`-- project2/
    |-- README.md
    |-- views.py                      # Training, SVG tree builder, Rashomon, ALE/PDP/M-Plot, counterfactuals
    |-- tests.py                      # 16 automated unit tests covering all components
    |-- urls.py
    |-- templates/project2/index.html # Interactive explainability dashboard
    `-- static/project2/style.css     # Tooltip, SVG glow, and interactive layout styling
```

## Run the Project

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python manage.py runserver
```

Open <http://127.0.0.1:8000/project2/>.

## Check and Test

Run all unit tests:

```powershell
$env:MPLCONFIGDIR = ".matplotlib_cache"
python manage.py test project2 --verbosity 1
```

The test suite contains 16 automated unit tests covering dataset cleaning, readable feature labels, decision tree metrics, candidate selection, M-Plots, ALE/PDP, tree path tracing, rule list extraction, MAD categorical penalties, Rashomon ratio calculations, logistic regression odds ratios, and page template rendering.

## Suggested Evaluation Path

1. Open the Project 2 page.
2. Review the data-transparency panel for target, classes, features, encoding, and removed missing rows.
3. Inspect the decision tree, accuracy, number of leaves, and rule list.
4. Move the lambda slider and apply the preference to compare accuracy against model simplicity.
5. Switch between decision tree and logistic regression to compare different interpretable model families.
6. Select a penguin and target species to generate counterfactual examples.
7. Choose a numeric feature and update PDP, M-Plot, and ALE feature-effect plots.
