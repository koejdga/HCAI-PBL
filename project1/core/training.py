import os
import time
import joblib
import pickle
import uuid
import numpy as np
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, balanced_accuracy_score, f1_score

from .dataset import numeric_values, categorical_values, get_column_type, configure_dataset

TASK_OPTIONS = [
    {"value": "classification", "label": "Classification"},
    {"value": "regression", "label": "Regression"},
]

def calculate_average_target(dataset):
    target = dataset["target"]
    target_column = dataset["columns"][target]
    if target_column["type"] == "numeric":
        return float(np.nanmean(numeric_values(target_column["values"])))
    return None

def infer_task_type(dataset):
    target_column = dataset["columns"][dataset["target"]]
    if target_column["type"] == "categorical":
        return "classification"
    values = numeric_values(target_column["values"])
    values = values[~np.isnan(values)]
    unique_values = np.unique(values)
    integer_coded = np.all(np.isclose(unique_values, np.round(unique_values)))
    if integer_coded and len(unique_values) <= 10 and len(unique_values) <= len(values) / 2:
        return "classification"
    return "regression"

def get_effective_task_type(dataset):
    return dataset.get("task_type") or infer_task_type(dataset)

def validate_task_type(dataset, task_type):
    valid_task_types = {option["value"] for option in TASK_OPTIONS}
    if task_type not in valid_task_types:
        return infer_task_type(dataset)
    target_column = dataset["columns"][dataset["target"]]
    if task_type == "regression" and target_column["type"] != "numeric":
        raise ValueError("Regression requires a numeric target column.")
    return task_type

def training_model_options(task_type):
    if task_type == "classification":
        return [
            {"value": "logistic_regression", "label": "Logistic Regression", "hyperparameter": "C"},
            {"value": "knn_classifier", "label": "K-Nearest Neighbors", "hyperparameter": "n_neighbors"},
            {"value": "svc", "label": "Support Vector Machines", "hyperparameter": "C"},
            {"value": "random_forest_classifier", "label": "Random Forest", "hyperparameter": "n_estimators"},
            {"value": "gaussian_nb", "label": "Gaussian Naive Bayes", "hyperparameter": None},
        ]
    return [
        {"value": "linear_regression", "label": "Linear Regression", "hyperparameter": None},
        {"value": "ridge_regression", "label": "Ridge Regression", "hyperparameter": "alpha"},
        {"value": "lasso_regression", "label": "Lasso Regression", "hyperparameter": "alpha"},
        {"value": "knn_regressor", "label": "K-Nearest Neighbors", "hyperparameter": "n_neighbors"},
        {"value": "svr", "label": "Support Vector Machines", "hyperparameter": "C"},
        {"value": "random_forest_regressor", "label": "Random Forest", "hyperparameter": "n_estimators"},
    ]

def get_model_choice(model_options, selected_model):
    if selected_model:
        for option in model_options:
            if option["value"] == selected_model:
                return option
    return model_options[0]

def clean_numeric_value(value):
    return np.nan if value == "" else float(value)

def get_active_features(dataset):
    features = dataset.get("features", [])
    return features

def get_kept_row_indexes(dataset):
    active_features = get_active_features(dataset)
    imputation_strategies = dataset.get("imputation_strategies", {})
    drop_rows_features = [f for f in active_features if imputation_strategies.get(f) == "drop_rows"]
    
    columns = dataset["columns"]
    row_count = dataset["row_count"]
    
    kept = []
    for r in range(row_count):
        if not any(columns[f]["values"][r] == "" for f in drop_rows_features):
            kept.append(r)
    return kept

def build_training_arrays(dataset):
    active_features = get_active_features(dataset)
    if not active_features:
        raise ValueError("Please keep at least one active feature column for training.")

    columns = dataset["columns"]
    numeric_feature_indexes = []
    categorical_feature_indexes = []
    
    kept_indexes = get_kept_row_indexes(dataset)
    if not kept_indexes:
        raise ValueError("Dropping rows for missing values removed the entire dataset. Please choose another strategy.")

    rows = []
    for row_index in kept_indexes:
        row = []
        for feature_index, feature in enumerate(active_features):
            column = columns[feature]
            value = column["values"][row_index]
            if column["type"] == "numeric":
                row.append(clean_numeric_value(value))
                if feature_index not in numeric_feature_indexes:
                    numeric_feature_indexes.append(feature_index)
            else:
                row.append(value if value != "" else "")
                if feature_index not in categorical_feature_indexes:
                    categorical_feature_indexes.append(feature_index)
        rows.append(row)

    return rows, numeric_feature_indexes, categorical_feature_indexes

def build_target_values(dataset, task_type):
    target_column = dataset["columns"][dataset["target"]]
    kept_indexes = get_kept_row_indexes(dataset)
    
    filtered_values = [target_column["values"][i] for i in kept_indexes]
    
    if task_type == "regression":
        values = numeric_values(filtered_values)
        if np.isnan(values).any():
            raise ValueError("Regression targets cannot contain missing values.")
        return values

    if any(v == "" for v in filtered_values):
        raise ValueError("Classification targets cannot contain missing values.")

    return np.array(categorical_values(filtered_values))

def build_preprocessor(dataset, numeric_feature_indexes, categorical_feature_indexes):
    features = get_active_features(dataset)
    imputation_strategies = dataset.get("imputation_strategies", {})
    
    transformers = []
    
    for idx in numeric_feature_indexes:
        feat_name = features[idx]
        strategy = imputation_strategies.get(feat_name, "median")

        if strategy in {"constant", "drop_column"}:
            strategy = "median"
        elif strategy not in ["mean", "median", "most_frequent"]:
            strategy = "median"

        imputer = SimpleImputer(strategy=strategy)
            
        feat_pipeline = Pipeline(
            [
                ("imputer", imputer),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append((f"numeric_{feat_name}", feat_pipeline, [idx]))

    for idx in categorical_feature_indexes:
        feat_name = features[idx]
        strategy = imputation_strategies.get(feat_name, "most_frequent")

        if strategy in {"constant", "drop_column"}:
            strategy = "most_frequent"
        elif strategy not in ["most_frequent"]:
            strategy = "most_frequent"

        imputer = SimpleImputer(strategy=strategy, missing_values="")
            
        feat_pipeline = Pipeline(
            [
                ("imputer", imputer),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            (f"categorical_{feat_name}", feat_pipeline, [idx])
        )

    return ColumnTransformer(transformers)

def model_candidates(model_name, train_size=None):
    max_neighbors = max(1, train_size or 7)
    
    if model_name == "logistic_regression":
        return [
            ("C=0.1", LogisticRegression(C=0.1, max_iter=1000)),
            ("C=1", LogisticRegression(C=1, max_iter=1000)),
            ("C=10", LogisticRegression(C=10, max_iter=1000)),
        ]
    if model_name == "knn_classifier":
        neighbor_values = [k for k in (3, 5, 7) if k <= max_neighbors]
        if not neighbor_values:
            neighbor_values = [1]
        return [
            (f"k={k}", KNeighborsClassifier(n_neighbors=k)) for k in neighbor_values
        ]
    if model_name == "svc":
        return [
            ("C=0.1", SVC(C=0.1)),
            ("C=1", SVC(C=1)),
            ("C=10", SVC(C=10)),
        ]
    if model_name == "random_forest_classifier":
        return [
            (
                "n_estimators=50",
                RandomForestClassifier(n_estimators=50, random_state=42),
            ),
            (
                "n_estimators=100",
                RandomForestClassifier(n_estimators=100, random_state=42),
            ),
            (
                "n_estimators=200",
                RandomForestClassifier(n_estimators=200, random_state=42),
            ),
        ]
    if model_name == "gaussian_nb":
        return [("default", GaussianNB())]

    if model_name == "linear_regression":
        return [("default", LinearRegression())]
    if model_name == "ridge_regression":
        return [
            ("alpha=0.1", Ridge(alpha=0.1)),
            ("alpha=1", Ridge(alpha=1)),
            ("alpha=10", Ridge(alpha=10)),
        ]
    if model_name == "lasso_regression":
        return [
            ("alpha=0.1", Lasso(alpha=0.1, max_iter=2000)),
            ("alpha=1", Lasso(alpha=1, max_iter=2000)),
            ("alpha=10", Lasso(alpha=10, max_iter=2000)),
        ]
    if model_name == "knn_regressor":
        neighbor_values = [k for k in (3, 5, 7) if k <= max_neighbors]
        if not neighbor_values:
            neighbor_values = [1]
        return [
            (f"k={k}", KNeighborsRegressor(n_neighbors=k)) for k in neighbor_values
        ]
    if model_name == "svr":
        return [
            ("C=0.1", SVR(C=0.1)),
            ("C=1", SVR(C=1)),
            ("C=10", SVR(C=10)),
        ]
    if model_name == "random_forest_regressor":
        return [
            (
                "n_estimators=50",
                RandomForestRegressor(n_estimators=50, random_state=42),
            ),
            (
                "n_estimators=100",
                RandomForestRegressor(n_estimators=100, random_state=42),
            ),
            (
                "n_estimators=200",
                RandomForestRegressor(n_estimators=200, random_state=42),
            ),
        ]

    raise ValueError("Please select a valid model.")

def build_baseline_comparison(task_type, model_score, baseline_score):
    difference = float(model_score) - float(baseline_score)
    tolerance = 0.0001
    
    if difference > tolerance:
        status = "better"
        headline = "Model beats the baseline"
        message = (
            "The trained model found more predictive signal than the naive reference "
            "on this test split."
        )
    elif difference < -tolerance:
        status = "worse"
        headline = "Model is below the baseline"
        message = (
            "The naive reference performed better on this test split. Review the "
            "data, selected features, model, and split before relying on it."
        )
    else:
        status = "similar"
        headline = "Model is similar to the baseline"
        message = (
            "The trained model did not show a meaningful improvement over the naive "
            "reference on this test split."
        )

    if task_type == "classification":
        difference_label = f"{difference * 100:+.2f} percentage points"
    else:
        difference_label = f"{difference:+.4f} R2"

    return {
        "difference": round(difference, 4),
        "difference_label": difference_label,
        "status": status,
        "headline": headline,
        "message": message,
    }

def format_metric_value(task_type, score):
    if task_type == "classification":
        return f"{float(score) * 100:.2f}%"
    return f"{float(score):.4f}"

def get_custom_model_candidates(model_name, selected_params, train_size=None):
    if not selected_params:
        return model_candidates(model_name, train_size)
        
    if model_name in ["logistic_regression", "svc", "svr", "ridge_regression", "lasso_regression"]:
        try:
            values = [float(v) for v in selected_params]
        except ValueError:
            values = [0.1, 1.0, 10.0]
            
        if model_name == "logistic_regression":
            return [(f"C={val}", LogisticRegression(C=val, max_iter=1000)) for val in values]
        elif model_name == "svc":
            return [(f"C={val}", SVC(C=val)) for val in values]
        elif model_name == "svr":
            return [(f"C={val}", SVR(C=val)) for val in values]
        elif model_name == "ridge_regression":
            return [(f"alpha={val}", Ridge(alpha=val)) for val in values]
        elif model_name == "lasso_regression":
            return [(f"alpha={val}", Lasso(alpha=val, max_iter=2000)) for val in values]
            
    elif model_name in ["knn_classifier", "knn_regressor"]:
        max_neighbors = max(1, train_size or 7)
        try:
            values = [int(v) for v in selected_params if int(v) <= max_neighbors]
            if not values:
                values = [1]
        except ValueError:
            values = [3, 5, 7]
            values = [k for k in values if k <= max_neighbors] or [1]
            
        if model_name == "knn_classifier":
            return [(f"k={val}", KNeighborsClassifier(n_neighbors=val)) for val in values]
        elif model_name == "knn_regressor":
            return [(f"k={val}", KNeighborsRegressor(n_neighbors=val)) for val in values]
            
    elif model_name in ["random_forest_classifier", "random_forest_regressor"]:
        try:
            values = [int(v) for v in selected_params]
        except ValueError:
            values = [50, 100, 200]
            
        if model_name == "random_forest_classifier":
            return [(f"n_estimators={val}", RandomForestClassifier(n_estimators=val, random_state=42)) for val in values]
        elif model_name == "random_forest_regressor":
            return [(f"n_estimators={val}", RandomForestRegressor(n_estimators=val, random_state=42)) for val in values]
            
    return model_candidates(model_name, train_size)

def train_model(dataset, model_name, test_size_percent, selected_params=None, enable_poly=False):
    task_type = get_effective_task_type(dataset)
    X, numeric_indexes, categorical_indexes = build_training_arrays(dataset)
    y = build_target_values(dataset, task_type)

    test_size = float(test_size_percent) / 100
    stratify = y if task_type == "classification" and len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    if task_type == "classification":
        baseline_name = "Most-frequent class"
        baseline_estimator = DummyClassifier(strategy="most_frequent")
    else:
        baseline_name = "Training-target mean"
        baseline_estimator = DummyRegressor(strategy="mean")

    baseline_pipeline = Pipeline(
        [
            (
                "preprocess",
                build_preprocessor(dataset, numeric_indexes, categorical_indexes),
            ),
            ("model", baseline_estimator),
        ]
    )
    
    start_time = time.time()
    baseline_pipeline.fit(X_train, y_train)
    baseline_training_time = time.time() - start_time
    baseline_predictions = baseline_pipeline.predict(X_test)

    if task_type == "classification":
        baseline_score = accuracy_score(y_test, baseline_predictions)
        baseline_bal_acc = balanced_accuracy_score(y_test, baseline_predictions)
        baseline_f1 = f1_score(y_test, baseline_predictions, average="weighted", zero_division=0)
        baseline_rmse = None
    else:
        baseline_score = r2_score(y_test, baseline_predictions)
        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))

    rows = []
    best_row = None
    best_pipeline = None
    higher_is_better = True
    
    candidates = get_custom_model_candidates(model_name, selected_params, len(X_train))
    
    for parameter_label, estimator in candidates:
        steps = [
            (
                "preprocess",
                build_preprocessor(dataset, numeric_indexes, categorical_indexes),
            )
        ]
        if enable_poly and model_name in ["linear_regression", "logistic_regression", "ridge_regression", "lasso_regression"]:
            steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
            
        steps.append(("model", estimator))
        pipeline = Pipeline(steps)
        
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        predictions = pipeline.predict(X_test)
        
        if task_type == "classification":
            score = accuracy_score(y_test, predictions)
            bal_acc = balanced_accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
            rmse = None
        else:
            score = r2_score(y_test, predictions)
            bal_acc = None
            f1 = None
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            
        row = {
            "parameter": parameter_label,
            "score": score,
            "display_score": format_metric_value(task_type, score),
            "display_balanced_accuracy": format_metric_value("classification", bal_acc) if bal_acc is not None else None,
            "display_f1_score": format_metric_value("classification", f1) if f1 is not None else None,
            "display_rmse": format_metric_value("regression", rmse) if rmse is not None else None,
            "display_training_time": f"{training_time * 1000:.2f} ms",
        }
        
        rows.append(row)
        
        if best_row is None:
            best_row = row
            best_pipeline = pipeline
        else:
            is_better = score > best_row["score"] if higher_is_better else score < best_row["score"]
            if is_better:
                best_row = row
                best_pipeline = pipeline

    comparison = build_baseline_comparison(task_type, best_row["score"], baseline_score)

    # Save best pipeline
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    joblib_path = os.path.join(settings.MEDIA_ROOT, f"best_pipeline_{uuid.uuid4().hex}.joblib")
    pkl_path = os.path.join(settings.MEDIA_ROOT, f"best_pipeline_{uuid.uuid4().hex}.pkl")
    
    joblib.dump(best_pipeline, joblib_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(best_pipeline, f)

    return {
        "model_value": model_name,
        "test_size_percent": test_size_percent,
        "enable_poly": enable_poly,
        "selected_params": selected_params,
        "task_type": task_type,
        "metric": "Accuracy" if task_type == "classification" else "R2 Score",
        "rows": rows,
        "best": best_row,
        "baseline": {
            "name": baseline_name,
            "score": baseline_score,
            "balanced_accuracy": baseline_bal_acc if task_type == "classification" else None,
            "f1": baseline_f1 if task_type == "classification" else None,
            "rmse": baseline_rmse if task_type == "regression" else None,
            "display_score": format_metric_value(task_type, baseline_score),
            "display_balanced_accuracy": format_metric_value("classification", baseline_bal_acc) if task_type == "classification" else None,
            "display_f1_score": format_metric_value("classification", baseline_f1) if task_type == "classification" else None,
            "display_rmse": format_metric_value("regression", baseline_rmse) if task_type == "regression" else None,
            "display_training_time": f"{baseline_training_time * 1000:.2f} ms",
        },
        "comparison": comparison,
        "joblib_path": joblib_path,
        "pkl_path": pkl_path,
    }

def train_all_and_compare(dataset, test_size_percent):
    task_type = get_effective_task_type(dataset)
    options = training_model_options(task_type)
    results = []
    
    for option in options:
        try:
            res = train_model(dataset, option["value"], test_size_percent)
            results.append({
                "model_name": option["label"],
                "model_value": option["value"],
                "best_param": res["best"]["parameter"],
                "score": res["best"]["score"],
                "display_score": res["best"]["display_score"],
                "display_balanced_accuracy": res["best"].get("display_balanced_accuracy"),
                "display_f1_score": res["best"].get("display_f1_score"),
                "display_rmse": res["best"].get("display_rmse"),
                "display_training_time": res["best"].get("display_training_time"),
                "baseline_score": res["baseline"]["score"],
                "display_baseline": res["baseline"]["display_score"],
                "metric": res["metric"],
                "comparison": res["comparison"],
            })
        except Exception:
            pass

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def get_task_source_note(dataset, task_type):
    inferred_task_type = dataset.get("inferred_task_type") or infer_task_type(dataset)
    if task_type == inferred_task_type:
        return "Matches the app's automatic type detection."
    return f"User override. The app initially detected {inferred_task_type}."

def selected_features_from_options(feature_options, selected_feature, x_feature, y_feature):
    if feature_options:
        selected_feature = selected_feature or feature_options[0]
        x_feature = x_feature or feature_options[0]
    if len(feature_options) >= 2:
        y_feature = y_feature or feature_options[1]
    return selected_feature, x_feature, y_feature

def build_reproduction_script_content(
    dataset_name,
    target_col,
    features,
    test_size_percent,
    model_value=None,
    best_param="default",
    enable_poly=False,
    task_type="classification",
    model_name=None,
):
    model_value = model_value or model_name
    test_size = float(test_size_percent) / 100
    
    model_imports = {
        "logistic_regression": "from sklearn.linear_model import LogisticRegression",
        "linear_regression": "from sklearn.linear_model import LinearRegression",
        "ridge_regression": "from sklearn.linear_model import Ridge",
        "lasso_regression": "from sklearn.linear_model import Lasso",
        "knn_classifier": "from sklearn.neighbors import KNeighborsClassifier",
        "knn_regressor": "from sklearn.neighbors import KNeighborsRegressor",
        "svc": "from sklearn.svm import SVC",
        "svr": "from sklearn.svm import SVR",
        "random_forest_classifier": "from sklearn.ensemble import RandomForestClassifier",
        "random_forest_regressor": "from sklearn.ensemble import RandomForestRegressor",
        "gaussian_nb": "from sklearn.naive_bayes import GaussianNB",
    }
    
    param_kwargs = ""
    if "=" in best_param:
        name, val = best_param.split("=")
        param_kwargs = f"{name}={val}"
    
    model_constructors = {
        "logistic_regression": f"LogisticRegression({param_kwargs or 'C=1.0'}, max_iter=1000)",
        "linear_regression": "LinearRegression()",
        "ridge_regression": f"Ridge({param_kwargs or 'alpha=1.0'})",
        "lasso_regression": f"Lasso({param_kwargs or 'alpha=1.0'}, max_iter=2000)",
        "knn_classifier": f"KNeighborsClassifier({param_kwargs or 'n_neighbors=5'})",
        "knn_regressor": f"KNeighborsRegressor({param_kwargs or 'n_neighbors=5'})",
        "svc": f"SVC({param_kwargs or 'C=1.0'})",
        "svr": f"SVR({param_kwargs or 'C=1.0'})",
        "random_forest_classifier": f"RandomForestClassifier({param_kwargs or 'n_estimators=100'}, random_state=42)",
        "random_forest_regressor": f"RandomForestRegressor({param_kwargs or 'n_estimators=100'}, random_state=42)",
        "gaussian_nb": "GaussianNB()",
    }

    model_import = model_imports.get(model_value, "from sklearn.linear_model import LogisticRegression")
    model_constructor = model_constructors.get(model_value, "LogisticRegression()")

    script = f'''# Python Reproduction Script for {dataset_name}
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder{', PolynomialFeatures' if enable_poly else ''}
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score, mean_squared_error
{model_import}

# 1. Load the dataset
# Update the path to your CSV file if necessary
df = pd.read_csv("{dataset_name}")

# Target and selected features
target_col = "{target_col}"
feature_cols = {features}

# 2. Separate features and target
X = df[feature_cols]
y = df[target_col]

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# 3. Train-Test Split
test_size = {test_size}
{'stratify = y if len(np.unique(y)) > 1 else None' if task_type == 'classification' else 'stratify = None'}
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=stratify
)

# 4. Preprocessing Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Build pipeline steps
steps = [('preprocessor', preprocessor)]
{f"steps.append(('poly', PolynomialFeatures(degree=2, include_bias=False)))" if enable_poly else "# No polynomial features"}
steps.append(('model', {model_constructor}))

pipeline = Pipeline(steps)

# 5. Fit the model
print("Training model...")
start_time = time.time()
pipeline.fit(X_train, y_train)
elapsed = time.time() - start_time
print(f"Training completed in {{elapsed * 1000:.2f}} ms")

# 6. Evaluate
predictions = pipeline.predict(X_test)
print("\\n--- Performance Metrics ---")
if "{task_type}" == "classification":
    acc = accuracy_score(y_test, predictions)
    bal_acc = balanced_accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    print(f"Accuracy: {{acc * 100:.2f}}%")
    print(f"Balanced Accuracy: {{bal_acc * 100:.2f}}%")
    print(f"F1-Score (Weighted): {{f1 * 100:.2f}}%")
else:
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"R2 Score: {{r2:.4f}}")
    print(f"RMSE: {{rmse:.4f}}")
'''
    return script
