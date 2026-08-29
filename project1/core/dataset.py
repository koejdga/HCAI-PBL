import csv
import io
import numpy as np

MISSING_MARKERS = {"", "na", "n/a", "nan", "null", "none", "?"}

def normalize_cell(value):
    value = value.strip()
    if value.lower() in MISSING_MARKERS:
        return ""
    return value

def get_column_type(values):
    non_empty_values = [value for value in values if value != ""]
    if not non_empty_values:
        return "categorical"
    try:
        for value in non_empty_values:
            float(value)
    except ValueError:
        return "categorical"
    return "numeric"

def numeric_values(values):
    return np.array([float(value) if value != "" else np.nan for value in values])

def categorical_values(values):
    return [value if value != "" else "(missing)" for value in values]

def top_categories(values, limit=12):
    categories, counts = np.unique(categorical_values(values), return_counts=True)
    order = np.argsort(counts)[::-1]
    return categories[order][:limit]

def encoded_column(column):
    if column["type"] == "numeric":
        values = numeric_values(column["values"])
        if np.isnan(values).any():
            fill_value = np.nanmean(values)
            values = np.nan_to_num(values, nan=fill_value)
        return values
    values = categorical_values(column["values"])
    categories = {value: index for index, value in enumerate(sorted(set(values)))}
    return np.array([categories[value] for value in values], dtype=float)

def copy_columns(columns):
    return {
        name: {"values": list(column["values"]), "type": column["type"]}
        for name, column in columns.items()
    }

def parse_csv_dataset(uploaded_file):
    decoded_file = uploaded_file.read().decode("utf-8-sig")
    reader = csv.reader(io.StringIO(decoded_file))
    rows = list(reader)
    if len(rows) < 2:
        raise ValueError("The CSV must contain a header row and at least one data row.")
    column_names = [column.strip() for column in rows[0]]
    if len(column_names) < 2:
        raise ValueError("The CSV must contain at least one feature column and one target column.")
    if any(name == "" for name in column_names):
        raise ValueError("Every CSV column must have a name.")
    if len(set(column_names)) != len(column_names):
        raise ValueError("CSV column names must be unique.")
    
    raw_rows = []
    for row_number, row in enumerate(rows[1:], start=2):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != len(column_names):
            raise ValueError(f"Row {row_number} has {len(row)} values, but the header has {len(column_names)} columns.")
        raw_rows.append([normalize_cell(value) for value in row])
        
    if not raw_rows:
        raise ValueError("No data rows found in the CSV.")
        
    columns = {}
    for index, name in enumerate(column_names):
        values = [row[index] for row in raw_rows]
        columns[name] = {"values": values, "type": get_column_type(values)}
        
    dataset = {
        "column_names": column_names,
        "default_target": column_names[-1],
        "columns": copy_columns(columns),
        "source_columns": copy_columns(columns),
        "row_count": len(raw_rows),
        "source_row_count": len(raw_rows),
        "drop_missing_rows": False,
        "dropped_row_count": 0,
    }
    return configure_dataset(dataset)

def is_likely_id_column(dataset, column_name):
    lowered_name = column_name.lower().replace("-", "_").replace(" ", "_")
    if lowered_name in {"id", "rowid", "row_id", "index"} or lowered_name.endswith("_id"):
        return True
    values = [value for value in dataset["columns"][column_name]["values"] if value != ""]
    if len(values) != dataset["row_count"]:
        return False
    try:
        numeric_values_for_column = [float(value) for value in values]
    except ValueError:
        return False
    if len(set(numeric_values_for_column)) != dataset["row_count"]:
        return False
    ordered = sorted(numeric_values_for_column)
    one_based = [float(index) for index in range(1, dataset["row_count"] + 1)]
    zero_based = [float(index) for index in range(dataset["row_count"])]
    return ordered == one_based or ordered == zero_based

def suggested_excluded_features(dataset, target):
    return [
        column_name
        for column_name in dataset["column_names"]
        if column_name != target and is_likely_id_column(dataset, column_name)
    ]

def restore_source_rows(dataset):
    if "source_columns" not in dataset:
        dataset["source_columns"] = copy_columns(dataset["columns"])
        dataset["source_row_count"] = dataset["row_count"]
    dataset["columns"] = copy_columns(dataset["source_columns"])
    dataset["row_count"] = dataset["source_row_count"]
    dataset["dropped_row_count"] = 0
    return dataset

def apply_missing_row_drop(dataset):
    kept_indexes = []
    for row_index in range(dataset["source_row_count"]):
        has_missing = any(
            dataset["columns"][column_name]["values"][row_index] == ""
            for column_name in dataset["column_names"]
        )
        if not has_missing:
            kept_indexes.append(row_index)
    if not kept_indexes:
        raise ValueError("Dropping missing rows would remove the whole dataset.")
    for column in dataset["columns"].values():
        column["values"] = [column["values"][index] for index in kept_indexes]
        column["type"] = get_column_type(column["values"])
    dataset["row_count"] = len(kept_indexes)
    dataset["dropped_row_count"] = dataset["source_row_count"] - len(kept_indexes)
    return dataset

def configure_dataset(dataset, target=None, features=None, task_type=None, drop_missing_rows=None):
    from .training import infer_task_type, validate_task_type
    dataset = restore_source_rows(dataset)
    column_names = dataset["column_names"]
    target = target or dataset.get("target") or dataset.get("default_target")
    if target not in column_names:
        raise ValueError("Please select a valid target column.")
    available_features = [name for name in column_names if name != target]
    suggested_exclusions = suggested_excluded_features(dataset, target)
    
    if features is None:
        existing_features = dataset.get("features", [])
        if existing_features:
            selected_features = [f for f in existing_features if f in available_features]
        else:
            selected_features = [f for f in available_features if f not in suggested_exclusions]
    else:
        selected_features = [f for f in features if f in available_features]
        
    if not selected_features:
        raise ValueError("Please keep at least one feature column for training.")
        
    if drop_missing_rows is None:
        drop_missing_rows = dataset.get("drop_missing_rows", False)
        
    dataset["target"] = target
    dataset["features"] = selected_features
    dataset["excluded_features"] = [f for f in available_features if f not in selected_features]
    dataset["suggested_excluded_features"] = suggested_exclusions
    dataset["inferred_task_type"] = infer_task_type(dataset)
    dataset["task_type"] = validate_task_type(dataset, task_type)
    dataset["drop_missing_rows"] = bool(drop_missing_rows)
    if dataset["drop_missing_rows"]:
        dataset = apply_missing_row_drop(dataset)
    return dataset

def count_missing_for_column(dataset, column_name):
    return sum(1 for value in dataset["columns"][column_name]["values"] if value == "")

def unique_non_missing_count(dataset, column_name):
    values = [value for value in dataset["columns"][column_name]["values"] if value != ""]
    return len(set(values))

def build_column_config_options(dataset):
    options = []
    target_name = dataset.get("target")
    col_vectors = {}
    for name in dataset["column_names"]:
        col = dataset["columns"][name]
        if col["type"] == "numeric":
            raw_vals = []
            for v in col["values"]:
                if v not in MISSING_MARKERS and v != "":
                    try:
                        raw_vals.append(float(v))
                    except ValueError:
                        pass
            mean_val = np.mean(raw_vals) if raw_vals else 0.0
            vals = [float(v) if (v not in MISSING_MARKERS and v != "") else mean_val for v in col["values"]]
            col_vectors[name] = np.array(vals)
        else:
            categories = list(set(col["values"]))
            cat_map = {cat: idx for idx, cat in enumerate(categories)}
            vals = [float(cat_map.get(v, 0)) for v in col["values"]]
            col_vectors[name] = np.array(vals)
            
    target_vec = col_vectors.get(target_name) if target_name else None
    for idx, name in enumerate(dataset["column_names"]):
        r = 0.0
        if target_name and name != target_name and target_vec is not None:
            vec = col_vectors.get(name)
            if vec is not None and len(vec) > 1 and np.std(vec) > 0 and np.std(target_vec) > 0:
                r = float(np.corrcoef(vec, target_vec)[0, 1])
        abs_r = abs(r)
        strength = "Strong" if abs_r > 0.5 else ("Moderate" if abs_r > 0.1 else "Low")
        direction = "Positive" if r >= 0 else "Negative"
        
        if strength == "Low":
            correlation_label = f"Low Relevance: {r:+.2f} [Suggested: Exclude]"
            suggest_exclude = True
        else:
            correlation_label = f"{strength} {direction}: {r:+.2f}"
            suggest_exclude = False
            
        options.append({
            "name": name,
            "type": dataset["columns"][name]["type"],
            "is_target": name == target_name,
            "is_feature": name in dataset["features"],
            "missing_count": count_missing_for_column(dataset, name),
            "unique_count": unique_non_missing_count(dataset, name),
            "likely_id": is_likely_id_column(dataset, name),
            "correlation_r": r,
            "abs_correlation_r": abs_r,
            "correlation_label": correlation_label,
            "suggest_exclude": suggest_exclude,
            "dataset_order": idx,
        })
    return options

def get_dataset_summary(dataset):
    target = dataset["target"]
    target_column = dataset["columns"][target]
    return {
        "rows": dataset["row_count"],
        "source_rows": dataset.get("source_row_count", dataset["row_count"]),
        "dropped_rows": dataset.get("dropped_row_count", 0),
        "drop_missing_rows": dataset.get("drop_missing_rows", False),
        "feature_count": len(dataset["features"]),
        "target": target,
        "target_type": target_column["type"],
    }

def count_missing_values(dataset):
    return sum(1 for col in dataset["columns"].values() for val in col["values"] if val == "")

def get_missing_column_summary(dataset, limit=4):
    missing_columns = []
    for name in dataset["column_names"]:
        missing_count = count_missing_for_column(dataset, name)
        if missing_count:
            missing_columns.append({"name": name, "count": missing_count})
    return missing_columns[:limit]

def get_target_source_note(dataset):
    default_target = dataset.get("default_target")
    if dataset["target"] == default_target:
        return "Defaulted to the last CSV column, as described in the project brief."
    return f"User-selected target. The CSV last column is {default_target}."

def get_class_counts(dataset):
    target_values = [
        val for val in categorical_values(dataset["columns"][dataset["target"]]["values"])
        if val != "(missing)"
    ]
    categories, counts = np.unique(target_values, return_counts=True)
    return dict(zip(categories, counts))

def build_quality_warnings(dataset, task_type):
    warnings = []
    target_missing = count_missing_for_column(dataset, dataset["target"])
    if target_missing:
        warnings.append({
            "title": "Target has missing values",
            "message": f"{target_missing} target rows are missing. Training will ask for a target without missing labels.",
        })
    if dataset.get("drop_missing_rows") and dataset.get("dropped_row_count", 0):
        warnings.append({
            "title": "Rows dropped for missing values",
            "message": f"{dataset['dropped_row_count']} rows were removed before visualization and training.",
        })
    missing_columns = get_missing_column_summary(dataset, limit=8)
    if missing_columns:
        columns = ", ".join(f"{col['name']} ({col['count']})" for col in missing_columns)
        warnings.append({
            "title": "Missing values detected",
            "message": f"Columns with missing values: {columns}.",
        })
    if dataset.get("suggested_excluded_features"):
        warnings.append({
            "title": "Likely ID columns excluded",
            "message": "Excluded by default: " + ", ".join(dataset["suggested_excluded_features"]) + ". Review the feature list if these columns are meaningful.",
        })
    if task_type == "classification":
        class_counts = get_class_counts(dataset)
        if class_counts:
            largest_class = max(class_counts.values())
            total = sum(class_counts.values())
            if total and largest_class / total >= 0.7:
                warnings.append({
                    "title": "Class imbalance",
                    "message": f"The largest class contains {largest_class} of {total} labeled rows.",
                })
    high_cardinality_features = [
        feat for feat in dataset["features"]
        if dataset["columns"][feat]["type"] == "categorical" and unique_non_missing_count(dataset, feat) > 20
    ]
    if high_cardinality_features:
        warnings.append({
            "title": "High-cardinality categorical features",
            "message": "These features have many categories: " + ", ".join(high_cardinality_features[:5]) + ".",
        })
    if dataset["row_count"] < 30:
        warnings.append({
            "title": "Small dataset",
            "message": "Model scores can be unstable with fewer than 30 rows.",
        })
    return warnings

def build_dataset_assumptions(dataset, task_type, test_size_percent):
    from .training import get_task_source_note
    target = dataset["target"]
    target_type = dataset["columns"][target]["type"]
    column_count = len(dataset["column_names"])
    total_cells = dataset["row_count"] * column_count
    missing_count = count_missing_values(dataset)
    missing_columns = get_missing_column_summary(dataset)
    
    if dataset.get("drop_missing_rows"):
        missing_detail = f"{missing_count} empty cells remain after dropping {dataset.get('dropped_row_count', 0)} rows."
        imputation_note = "Rows containing missing values are removed before visualization and training."
    elif missing_count:
        missing_detail = f"{missing_count} of {total_cells} cells are empty."
        imputation_note = "During training, numeric feature gaps use median imputation and categorical feature gaps use the most frequent value."
    else:
        missing_detail = "No empty cells were detected."
        imputation_note = "If missing feature values appear later, numeric gaps use median imputation and categorical gaps use the most frequent value."
        
    if task_type == "classification":
        task_detail = "The target looks categorical or compact integer-coded."
        target_note = "Classification targets are compared by class labels."
    else:
        task_detail = "The target looks continuous numeric."
        target_note = "Regression targets must be numeric and cannot be missing."
        
    return {
        "cards": [
            {
                "label": "Dataset",
                "value": f"{dataset['row_count']} rows",
                "detail": f"{len(dataset['features'])} selected features, {column_count} columns total." + (f" Dropped {dataset.get('dropped_row_count', 0)} of {dataset.get('source_row_count')} uploaded rows." if dataset.get("drop_missing_rows") else ""),
            },
            {"label": "Target", "value": target, "detail": f"{get_target_source_note(dataset)} Detected as {target_type}."},
            {"label": "Task", "value": task_type.title(), "detail": f"{task_detail} {get_task_source_note(dataset, task_type)}"},
            {"label": "Cleaning", "value": "Drop rows" if dataset.get("drop_missing_rows") else "Keep rows", "detail": missing_detail},
            {"label": "Evaluation", "value": f"{test_size_percent}% test split", "detail": "Models are scored on a held-out test set using random seed 42."},
        ],
        "notes": [
            "The app defaults to the last CSV column from the assignment, but the user can change it.",
            target_note,
            imputation_note,
            "These assumptions are shown so the user can review them before trusting model scores.",
        ],
        "missing_columns": missing_columns,
        "quality_warnings": build_quality_warnings(dataset, task_type),
    }

def build_audit_columns(dataset):
    audit_cols = []
    total_rows = dataset["row_count"]
    for col_name in dataset["column_names"]:
        col = dataset["columns"][col_name]
        col_type = col["type"]
        non_missing_vals = [v for v in col["values"] if v not in MISSING_MARKERS and v != ""]
        unique_vals = set(non_missing_vals)
        unique_count = len(unique_vals)
        missing_count = sum(1 for v in col["values"] if v in MISSING_MARKERS or v == "")
        missing_percentage = f"{(missing_count / total_rows) * 100:.1f}%" if total_rows > 0 else "0.0%"
        
        min_val = max_val = mean_val = median_val = None
        suggest_median = suggest_mode = False
        display_type = col_type
        
        if col_type == "numeric" and unique_count <= 10:
            display_type = "Low-cardinality discrete"
            suggest_mode = True
        elif col_type == "categorical":
            suggest_mode = True
            
        if col_type == "numeric":
            raw_vals = []
            for v in non_missing_vals:
                try:
                    raw_vals.append(float(v))
                except ValueError:
                    pass
            if raw_vals:
                raw_arr = np.array(raw_vals)
                min_val = float(np.min(raw_arr))
                max_val = float(np.max(raw_arr))
                mean_val = float(np.mean(raw_arr))
                median_val = float(np.median(raw_arr))
                std_val = float(np.std(raw_arr)) if len(raw_arr) > 1 else 0.0
                
                # 1.5 * IQR outlier detection
                q75, q25 = np.percentile(raw_arr, [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outliers = raw_arr[(raw_arr < lower_bound) | (raw_arr > upper_bound)]
                has_outliers = len(outliers) > 0
                if not suggest_mode:
                    suggest_median = has_outliers or (std_val > 0 and abs(mean_val - median_val) / std_val > 0.1)
                    
        suggested_strategy = "most_frequent" if (display_type == "categorical" or suggest_mode) else ("median" if suggest_median else "mean")
        impute_strategy = dataset.get("imputation_strategies", {}).get(col_name, suggested_strategy)
        
        audit_cols.append({
            "name": col_name,
            "type": display_type,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "median": median_val,
            "suggest_median": suggest_median,
            "suggest_mode": suggest_mode,
            "suggested_strategy": suggested_strategy,
            "impute_strategy": impute_strategy,
        })
    return audit_cols

def get_missing_rows_info(dataset):
    if not dataset:
        return {"count": 0, "percentage": "0.0%"}
    total_rows = dataset["row_count"]
    active_cols = list(dataset.get("features", []))
    target = dataset.get("target")
    if target and target not in active_cols:
        active_cols.append(target)
    rows_to_drop = 0
    for row_idx in range(total_rows):
        is_missing = False
        for col_name in active_cols:
            if col_name in dataset["columns"]:
                val = dataset["columns"][col_name]["values"][row_idx]
                if val in MISSING_MARKERS or val == "":
                    is_missing = True
                    break
        if is_missing:
            rows_to_drop += 1
    percentage = (rows_to_drop / total_rows) * 100 if total_rows > 0 else 0.0
    return {"count": rows_to_drop, "percentage": f"{percentage:.1f}%"}

def build_correlation_data(dataset):
    if not dataset:
        return None
    target_name = dataset.get("target")
    numeric_cols = [name for name in dataset["column_names"] if dataset["columns"][name]["type"] == "numeric" and name != target_name]
    if not numeric_cols:
        return None
    
    col_data = {}
    for col_name in dataset["column_names"]:
        col = dataset["columns"][col_name]
        if col["type"] == "numeric":
            raw_vals = [float(v) for v in col["values"] if v not in MISSING_MARKERS and v != ""]
            mean_val = np.mean(raw_vals) if raw_vals else 0.0
            vals = [float(v) if (v not in MISSING_MARKERS and v != "") else mean_val for v in col["values"]]
            col_data[col_name] = np.array(vals)
        else:
            categories = list(set(col["values"]))
            cat_map = {cat: idx for idx, cat in enumerate(categories)}
            vals = [float(cat_map.get(v, 0)) for v in col["values"]]
            col_data[col_name] = np.array(vals)
            
    if target_name and target_name not in col_data:
        col = dataset["columns"][target_name]
        categories = list(set(col["values"]))
        cat_map = {cat: idx for idx, cat in enumerate(categories)}
        vals = [float(cat_map.get(v, 0)) for v in col["values"]]
        col_data[target_name] = np.array(vals)
        
    matrix_rows = []
    for c1 in numeric_cols:
        cells = []
        vec1 = col_data[c1]
        for c2 in numeric_cols:
            vec2 = col_data[c2]
            r = float(np.corrcoef(vec1, vec2)[0, 1]) if (len(vec1) > 1 and np.std(vec1) > 0 and np.std(vec2) > 0) else 0.0
            bg_color = f"rgba(0, 141, 162, {r:.2f})" if r >= 0 else f"rgba(229, 62, 62, {abs(r):.2f})"
            text_color = "#ffffff" if abs(r) > 0.5 else "#2d3748"
            cells.append({"val": r, "bg_color": bg_color, "text_color": text_color})
        matrix_rows.append({"feature": c1, "cells": cells})
    return {"features": numeric_cols, "matrix_rows": matrix_rows}

def normalize_dataset_columns(dataset):
    if "source_columns" not in dataset:
        dataset["source_columns"] = copy_columns(dataset["columns"])
        dataset["source_row_count"] = dataset["row_count"]
    for cols in (dataset["source_columns"], dataset["columns"]):
        for col in cols.values():
            values = [normalize_cell(str(val)) for val in col["values"]]
            col["values"] = values
            col["type"] = get_column_type(values)
    return dataset

def ensure_dataset_configuration(dataset):
    if not dataset:
        return None
    dataset = normalize_dataset_columns(dataset)
    dataset.setdefault("default_target", dataset.get("target") or dataset["column_names"][-1])
    return configure_dataset(
        dataset,
        target=dataset.get("target"),
        features=dataset.get("features"),
        task_type=dataset.get("task_type"),
    )
