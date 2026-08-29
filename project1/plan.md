# Human-Centric AI — Project 1: Enhanced Supervised Learning & AutoML Interface
## Technical Specification & Implementation Blueprint

### 1. Architectural Philosophy & Dual-Track Workflow
The application must support two operational pathways accessible via a prominent toggle:
1. **AutoML Track (Full Automation):** Automatically executes data audits, preprocessing, benchmark training across compatible models, selects the best performer, and produces diagnostic reports[cite: 1].
2. **Manual Track (Human-in-the-Loop):** Gives users explicit control over imputation strategies, feature selection, model selection, hyperparameter sweeps (e.g., $C$), and polynomial expansions[cite: 1, 5].

---

### 2. UI Layout & Navigation Hierarchy
* **Global Navigation:** Remove the top horizontal bar[cite: 5]. Replace it with a responsive, structured left sidebar.
* **Sidebar Structure:**
  * **1. Data Management:** Preloaded Example Datasets, Custom CSV Upload, "Your Datasets" (Session/History Manager)[cite: 1, 5].
  * **2. Data Audit & EDA:** Quality Health Check, Correlation Heatmap, Feature Distribution & 2D Scatter Plots[cite: 1, 5].
  * **3. Feature Studio:** Relevance Bar Plot & Filtering Checkboxes[cite: 5].
  * **4. Model Studio:** Mode Switch (AutoML vs. Manual), Training Pipeline[cite: 1, 5].
  * **5. Diagnostics & History:** Learning Curves, Confusion Matrix/Residuals, Experiment Run Log[cite: 5].
  * **6. Exports:** Model Artifacts (.joblib), Reproducible Python Script (.py), Summary Report[cite: 1].
* **In-App Descriptions & Tooltips:** Every section, diagnostic chart, metric, and hyperparameter must include a human-readable explanation or expandable `(?)` information accordion explaining *why it matters* and *how to interpret it* (e.g., explaining baseline comparison, score metrics, and regularization parameter $C$)[cite: 1, 5].

---

### 3. Functional Modules & Implementation Details

#### Module A: Data Management & Built-in Datasets
* **Preloaded Datasets:** Provide a dropdown with classic datasets[cite: 1]:
  * *Classification:* Iris (Multiclass, 4 features)[cite: 1], Wine Recognition (Multiclass, 13 features), Breast Cancer Wisconsin (Binary, 30 features).
  * *Regression:* California Housing (8 features) or Diabetes (10 features).
* **CSV Upload:** Support custom scalar CSV files where the target column defaults to the last column (with user override capability)[cite: 1, 5].
* **"Your Datasets" Manager:** Store uploaded or selected datasets within the session so users can switch between them without re-uploading[cite: 5].

#### Module B: Data Audit & Smart Preprocessing
* **Missing Value Inspection:** Report exact count and percentage of missing values per feature[cite: 5].
* **Imputation Recommender:**
  * If column skewness is high ($|\text{skew}| > 1$), show: *"Recommended: Median (due to skewness/outliers)"*.
  * Otherwise: *"Recommended: Mean"*.
* **User Cleaning Controls:** Provide options to *Drop rows with missing values* or *Impute (Mean / Median / Constant)*[cite: 5].
* **Feature Encoding & Scaling:** Automatically bundle `OneHotEncoder(handle_unknown='ignore')` for categorical features and `StandardScaler` for continuous features inside a scikit-learn `Pipeline` to prevent data leakage[cite: 5].

#### Module C: Feature Relevance & Interactive Selection
* **Relevance Metric:** Compute statistical target relevance using **Mutual Information** (`mutual_info_classif` or `mutual_info_regression`) or **Target Correlation**[cite: 5].
* **Visual Relevance Chart:** Render a horizontal bar plot showing feature relevance ranking[cite: 5].
* **Sorting Toggle:** Allow users to sort features:
  1. *By Original CSV Order*[cite: 5].
  2. *By Descending Target Relevance*.
* **Interactive Feature Selection:**
  * Individual checkmarks for each feature[cite: 5].
  * Quick-action button: *"Select Top 10 Relevant Features"*.
  * Selected features are dynamically filtered before training[cite: 5].

#### Module D: Model Training & Execution Modes

**1. Manual Mode (User-Driven):**
* **Model Dropdown:** Based on the detected target type (Classification vs. Regression)[cite: 1, 5]:
  * *Classification:* Logistic Regression, k-NN, SVC, Random Forest, Gaussian Naive Bayes[cite: 5].
  * *Regression:* Ridge/Lasso Linear Regression, k-NN Regressor, SVR, Random Forest Regressor.
* **Pre-Run Heuristic Recommendations:** Before training, display advisory chips:
  * For small $N$ ($< 200$ rows): *"Recommended: Logistic Regression / SVM (Low risk of overfitting)"*[cite: 5].
  * For non-linear relationships: *"Recommended: Random Forest"*.
* **Hyperparameter Sweeps:** Allow the user to select parameter grids (e.g., $C \in \{0.1, 1, 10\}$ for Logistic/SVM or $k \in \{3, 5, 7\}$ for k-NN) and evaluate scores across test splits[cite: 1, 5].
* **Polynomial & Interaction Expansion:** If a Linear/Logistic model is selected, provide a checkbox: `[ ] Enable Polynomial & Interaction Features (Degree 2)`.

**2. AutoML Mode (Fully Automated):**
* **"Run All & Benchmark" Button:** Trains 4–5 compatible models on the identical train/test split[cite: 1, 5].
* **Leaderboard Table:**
  * Sortable columns: Model Name, Accuracy / $R^2$, Balanced Accuracy / RMSE, F1-Score, Training Time[cite: 1, 5].
  * Visual badge on the top-performing model[cite: 5].
  * 1-sentence analytical takeaway for every model (e.g., *"Random Forest captured complex interactions best"*, *"k-NN suffered from uninformative dimensions"*).

#### Module E: Diagnostics & Experiment Run History
* **Baseline Comparison:** Compare trained model performance against the simplest naive baseline (majority class for classification; mean target for regression)[cite: 5].
* **Overfitting / Underfitting Diagnostic:** Render a **Train vs. Validation Learning Curve** plotting cross-validation accuracy across training set sizes.
* **Detailed Diagnostics:**
  * *Classification:* Interactive Confusion Matrix heatmap (True vs. Predicted) and Precision-Recall/ROC curves.
  * *Regression:* Residuals vs. Predicted plot and Error distribution histogram.
* **Experiment Run History:** Maintain a session log table containing:
  * Run ID, Model Architecture, Hyperparameters ($C$, $k$), Active Features Count, Test Accuracy / Score, and Timestamp[cite: 5].
  * Action button: *"Clear History"*.

#### Module F: Exports & Reproducibility
* **Download Trained Pipeline (`.joblib`):** Download the complete fitted `sklearn.pipeline.Pipeline` (preprocessing + model) as an optimized `.joblib` binary.
* **Download Python Reproduction Script (`.py`):** Dynamically generate and export a clean, executable standalone `.py` script that reproduces the exact data ingestion, split ratio, imputation, scaling, feature subsetting, and tuned model hyperparameters configured in the UI[cite: 1, 5].
* **HTML/PDF Summary Report:** Provide a 1-click exportable summary of dataset statistics, benchmark rankings, and final model evaluation[cite: 1, 5].

---

### 4. Course Guardrails & Topic Boundary Protection
To preserve academic separation between coursework assignments, **strictly avoid** implementing:
* Tree structure leaf inspection or custom complexity regularization penalty ($\lambda \Omega(f)$) *(Project 2)*[cite: 4].
* Custom Counterfactual instance generators *(Project 2)*[cite: 4].
* Partial Dependence Plots (PDP) and Accumulated Local Effects (ALE) *(Project 2)*[cite: 4].
* Active learning query loops or human-AI deferral decision policies *(Project 3)*[cite: 3].
* Bradley-Terry preference elicitation rankings or recommender user studies *(Project 4)*[cite: 2].
