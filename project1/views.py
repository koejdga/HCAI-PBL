import os
from html import escape
from io import BytesIO

from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse, FileResponse
from django.conf import settings
from xhtml2pdf import pisa

from .core.dataset import (
    parse_csv_dataset,
    configure_dataset,
    get_dataset_summary,
    build_column_config_options,
    build_audit_columns,
    get_missing_rows_info,
    build_correlation_data,
    ensure_dataset_configuration,
    build_dataset_assumptions,
)
from .core.visualization import (
    save_overview_visualizations,
    save_feature_target_visualization,
    save_scatter_visualization,
    save_univariate_visualization,
    build_saved_visualizations,
    feature_target_key,
    scatter_key,
    univariate_key,
)
from .core.training import (
    TASK_OPTIONS,
    calculate_average_target,
    get_effective_task_type,
    training_model_options,
    train_model,
    train_all_and_compare,
    selected_features_from_options,
    build_reproduction_script_content,
    build_baseline_comparison,
    format_metric_value,
)

def index(request):
    return upload_csv(request)


def make_pdf_response(html_content, filename):
    pdf_buffer = BytesIO()
    result = pisa.CreatePDF(BytesIO(html_content.encode("utf-8")), pdf_buffer)
    if result.err:
        raise ValueError("Failed to generate PDF summary.")
    pdf_buffer.seek(0)
    response = FileResponse(pdf_buffer, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


def pdf_summary_report(request):
    dataset = request.session.get("project1_dataset")
    comparison_results = request.session.get("project1_comparison_results", [])

    if not dataset or not comparison_results:
        return HttpResponse("No comparison results found in session.", status=404)

    dataset_name = dataset.get("name") or "dataset"
    target_name = dataset.get("target") or "target"
    feature_names = dataset.get("features", [])
    task_type = get_effective_task_type(dataset)

    rows_html = []
    for index, row in enumerate(comparison_results, start=1):
        is_best = index == 1
        metric_cell = f"<td class='metric-cell accuracy-cell'><span class='{ 'score-good' if is_best else 'score-neutral' }'>{row.get('display_score', '')}</span></td>"
        extra_metrics = ""
        if task_type == "classification":
            extra_metrics = (
                f"<td class='metric-cell balanced-cell'>{row.get('display_balanced_accuracy', '')}</td>"
                f"<td class='metric-cell f1-cell'>{row.get('display_f1_score', '')}</td>"
            )
        else:
            extra_metrics = f"<td class='metric-cell rmse-cell'>{row.get('display_rmse', '')}</td>"

        comparison_status = row.get("comparison", {}).get("status", "similar")
        if comparison_status not in {"better", "similar", "worse"}:
            comparison_status = "similar"
        verdict_text = escape(str(row.get('comparison', {}).get('headline', '')))
        rows_html.append(
            f"<tr class='{ 'best-row' if is_best else '' }'>"
            f"<td class='rank-cell'>#{index}</td>"
            f"<td class='model-cell'>{escape(str(row.get('model_name', '')))}</td>"
            f"<td class='param-cell'>{escape(str(row.get('best_param', '')))}</td>"
            f"{metric_cell}"
            f"{extra_metrics}"
            f"<td class='time-cell'>{row.get('display_training_time', '')}</td>"
            f"<td class='verdict-cell'><span class='verdict-badge verdict-{comparison_status}'>{verdict_text}</span></td>"
            f"</tr>"
        )

    feature_items = "".join(f"<li>{escape(str(feature))}</li>" for feature in feature_names)
    if not feature_items:
        feature_items = "<li>No features selected</li>"

    metric_headers = (
        "<th class='accuracy-cell'>Accuracy</th><th class='balanced-cell'>Balanced Accuracy</th><th class='f1-cell'>F1-Score</th>"
        if task_type == "classification"
        else "<th class='accuracy-cell'>R² Score</th><th class='rmse-cell'>RMSE</th>"
    )
    column_widths = (
        """
        <col style="width: 5%;" />
        <col style="width: 18%;" />
        <col style="width: 17%;" />
        <col style="width: 10%;" />
        <col style="width: 14%;" />
        <col style="width: 10%;" />
        <col style="width: 10%;" />
        <col style="width: 16%;" />
        """
        if task_type == "classification"
        else """
        <col style="width: 5%;" />
        <col style="width: 21%;" />
        <col style="width: 19%;" />
        <col style="width: 12%;" />
        <col style="width: 12%;" />
        <col style="width: 12%;" />
        <col style="width: 19%;" />
        """
    )

    html_content = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>{escape(dataset_name)} Summary Report</title>
        <style>
          @page {{
            size: a4 landscape;
            margin: 1cm;
          }}
          body {{
            font-family: Helvetica, Arial, sans-serif;
            color: #1e2d3d;
            font-size: 11pt;
            line-height: 1.35;
            margin: 0;
            background: #ffffff;
          }}
          .main-title {{
            color: #0f2c4b;
            font-size: 22pt;
            font-weight: 700;
            margin: 0 0 18px 0;
            padding-bottom: 6px;
          }}
          .dataset-summary {{
            margin-bottom: 12px;
            font-size: 10pt;
            color: #1e2d3d;
          }}
          .meta-grid {{
            margin-bottom: 18px;
          }}
          .meta-row {{
            margin-bottom: 4px;
            font-size: 10.5pt;
            color: #1e2d3d;
          }}
          .feature-list {{
            margin: 4px 0 0 18px;
            padding: 0;
            list-style: disc;
            font-size: 10pt;
            color: #1e2d3d;
          }}
          .table-heading {{
            color: #0f2c4b;
            font-size: 13pt;
            font-weight: 700;
            margin: 18px 0 8px 0;
          }}
          .comparison-table-wrap {{
            width: 100%;
            border: 1px solid #d5e2ef;
            border-radius: 0;
            overflow: hidden;
            margin-top: 0;
            background: #ffffff;
          }}
          table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            margin: 0;
          }}
          thead th {{
            background-color: #f1f5f9;
            color: #1e2d3d;
            font-size: 9pt;
            font-weight: 700;
            padding: 5px 6px;
            text-align: left;
            border: 1px solid #d5e2ef;
            line-height: 1.2;
          }}
          tbody td {{
            padding: 5px 6px;
            border: 1px solid #d5e2ef;
            font-size: 9pt;
            color: #1e2d3d;
            vertical-align: middle;
            line-height: 1.2;
          }}
          tbody tr.best-row {{
            background-color: #eafaf1;
          }}
          .rank-cell {{
            font-weight: 700;
            color: #1e2d3d;
            white-space: nowrap;
            width: 5%;
          }}
          .model-cell {{
            width: 18%;
          }}
          .param-cell {{
            width: 17%;
          }}
          .accuracy-cell {{
            width: 10%;
          }}
          .balanced-cell {{
            width: 14%;
          }}
          .f1-cell {{
            width: 10%;
          }}
          .rmse-cell {{
            width: 12%;
          }}
          .time-cell {{
            width: 10%;
          }}
          .model-cell, .param-cell, .time-cell {{
            color: #1e2d3d;
          }}
          .metric-cell {{
            font-weight: 600;
            color: #1e2d3d;
          }}
          .score-good {{
            color: #16a34a;
            font-weight: 700;
          }}
          .score-neutral {{
            color: #1e2d3d;
            font-weight: 600;
          }}
          .verdict-cell {{
            text-align: left;
            width: 16%;
          }}
          .verdict-badge {{
            display: inline-block;
            color: #0f2c4b;
            padding: 2px 5px;
            font-size: 8.5pt;
            font-weight: 700;
            line-height: 1.2;
          }}
          .verdict-better {{
            background-color: #d0fdd4;
          }}
          .verdict-similar {{
            background-color: #fff9c8;
          }}
          .verdict-worse {{
            background-color: #ffe2e2;
          }}
          .note {{
            font-size: 9pt;
            color: #64748b;
            margin-top: 12px;
            display: block;
          }}
        </style>
      </head>
      <body>
        <div class="main-title">{escape(dataset_name)} — Model Comparison Summary</div>

        <div class="dataset-summary">
          <div class="meta-row"><strong>Dataset:</strong> {escape(dataset_name)}</div>
          <div class="meta-row"><strong>Target:</strong> {escape(target_name)}</div>
          <div class="meta-row"><strong>Features:</strong></div>
          <ul class="feature-list">{feature_items}</ul>
        </div>

        <div class="table-heading">Model Comparison Table</div>

        <div class="comparison-table-wrap">
          <table>
            <colgroup>
              {column_widths}
            </colgroup>
            <thead>
              <tr>
                <th class="rank-cell">Rank</th>
                <th class="model-cell">Model Algorithm</th>
                <th class="param-cell">Best Hyperparameter</th>
                {metric_headers}
                <th class="time-cell">Fit Time</th>
                <th class="verdict-cell">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows_html)}
            </tbody>
          </table>
        </div>
        <span class="note">* The best-performing model is highlighted in green.</span>
      </body>
    </html>
    """

    filename = f"{dataset_name.replace(' ', '_')}_comparison_summary.pdf"
    return make_pdf_response(html_content, filename)


def is_ajax_request(request):
    return (
        request.headers.get("x-requested-with") == "XMLHttpRequest"
        or request.headers.get("accept") == "application/json"
        or request.POST.get("ajax") == "1"
    )

def plot_payload(plot):
    payload = {
        "key": plot["key"],
        "title": plot["title"],
        "url": plot["url"],
    }
    if "feature" in plot:
        payload["feature"] = plot["feature"]
    if "x_feature" in plot:
        payload["x_feature"] = plot["x_feature"]
    if "y_feature" in plot:
        payload["y_feature"] = plot["y_feature"]
    return payload

def ajax_error(message, status=400):
    return JsonResponse({"status": "error", "message": message}, status=status)

def reset_project1_outputs(request):
    request.session["project1_feature_specs"] = []
    request.session["project1_scatter_specs"] = []
    request.session["project1_univariate_specs"] = []
    request.session["project1_training_result"] = None
    request.session["project1_comparison_results"] = []

def upload_csv(request):
    export_format = request.GET.get("export")
    if export_format:
        training_result = request.session.get("project1_training_result")
        if not training_result:
            return HttpResponse("No trained model result found in session.", status=404)
        
        if export_format in ["joblib", "pkl"]:
            file_key = "joblib_path" if export_format == "joblib" else "pkl_path"
            file_path = training_result.get(file_key)
            if not file_path or not os.path.exists(file_path):
                return HttpResponse("Trained model file not found on disk.", status=404)
            
            filename = f"trained_pipeline.{export_format}"
            response = FileResponse(open(file_path, "rb"), content_type="application/octet-stream")
            response["Content-Disposition"] = f'attachment; filename="{filename}"'
            return response
            
        elif export_format == "py":
            dataset = request.session.get("project1_dataset")
            dataset_name = dataset.get("name") if dataset else "dataset.csv"
            feature_specs = request.session.get("project1_feature_specs", [])
            target_col = dataset.get("target") if dataset else "target"
            
            script_content = build_reproduction_script_content(
                dataset_name=dataset_name,
                target_col=target_col,
                features=[f["name"] for f in feature_specs if f["is_feature"]],
                test_size_percent=training_result.get("test_size_percent", 20),
                model_value=training_result.get("model_value"),
                best_param=training_result.get("best", {}).get("parameter", "default"),
                enable_poly=training_result.get("enable_poly", False),
                task_type=training_result.get("task_type", "classification"),
            )
            
            response = HttpResponse(script_content, content_type="text/x-python")
            response["Content-Disposition"] = 'attachment; filename="reproduce_model.py"'
            return response

    dataset = None
    result = None
    error = None
    upload_error = None
    duplicate_message = None
    overview_visualizations = []
    feature_target_plots = []
    scatter_plots = []
    univariate_plots = []
    dataset_summary = None
    feature_options = []
    selected_feature = None
    selected_x_feature = None
    selected_y_feature = None
    task_type = None
    model_options = []
    selected_model = None
    test_size_percent = 20
    training_result = None
    column_config_options = []
    feature_specs = []
    scatter_specs = []
    univariate_specs = []
    highlight_key = None

    if request.method == "POST":
        action = request.POST.get("action", "upload")

        if action == "select_example":
            dataset_name = request.POST.get("dataset_name")
            if dataset_name in ["iris.csv", "diabetes.csv", "breast_cancer.csv", "wine.csv", "housing.csv", "auto-mpg.csv"]:
                file_path = os.path.join(settings.BASE_DIR, "project1", "example_datasets", dataset_name)
                try:
                    with open(file_path, "rb") as file:
                        dataset = parse_csv_dataset(file)
                    dataset["name"] = dataset_name
                    request.session["project1_dataset"] = dataset
                    reset_project1_outputs(request)

                    result = calculate_average_target(dataset)
                    dataset_summary = get_dataset_summary(dataset)
                    feature_options = dataset["features"]
                    task_type = get_effective_task_type(dataset)
                    model_options = training_model_options(task_type)
                    overview_visualizations = save_overview_visualizations(dataset)
                except Exception as e:
                    error = f"Error processing example dataset: {str(e)}"
            else:
                error = "Invalid example dataset selected."

        elif action == "select_uploaded":
            dataset_name = request.POST.get("dataset_name")
            uploaded_list = request.session.get("project1_uploaded_datasets", [])
            target_data = next((d for d in uploaded_list if d["name"] == dataset_name), None)
            if target_data:
                dataset = target_data["dataset"]
                dataset["name"] = dataset_name
                request.session["project1_dataset"] = dataset
                reset_project1_outputs(request)

                result = calculate_average_target(dataset)
                dataset_summary = get_dataset_summary(dataset)
                feature_options = dataset["features"]
                task_type = get_effective_task_type(dataset)
                model_options = training_model_options(task_type)
                overview_visualizations = save_overview_visualizations(dataset)
            else:
                error = "Selected dataset not found in your uploads."

        elif action == "upload":
            file = request.FILES.get("file")
            if file and file.name.lower().endswith(".csv"):
                try:
                    dataset = parse_csv_dataset(file)
                    dataset["name"] = file.name
                    request.session["project1_dataset"] = dataset
                    reset_project1_outputs(request)

                    result = calculate_average_target(dataset)
                    dataset_summary = get_dataset_summary(dataset)
                    feature_options = dataset["features"]
                    task_type = get_effective_task_type(dataset)
                    model_options = training_model_options(task_type)
                    overview_visualizations = save_overview_visualizations(dataset)

                    # Save to "Your datasets" in session
                    uploaded_list = request.session.get("project1_uploaded_datasets", [])
                    name = file.name
                    features_count = len(dataset["features"])
                    row_count = dataset["row_count"]
                    effective_type = get_effective_task_type(dataset)
                    info = f"{effective_type.capitalize()} target: {features_count} features, {row_count:,} rows"
                    # Deduplicate name in list
                    uploaded_list = [d for d in uploaded_list if d["name"] != name]
                    uploaded_list.append({
                        "name": name,
                        "info": info,
                        "dataset": dataset
                    })
                    request.session["project1_uploaded_datasets"] = uploaded_list

                except Exception as e:
                    error = f"Error processing file: {str(e)}"
            else:
                upload_error = "Please upload a CSV file."
        else:
            dataset = ensure_dataset_configuration(
                request.session.get("project1_dataset")
            )
            feature_specs = request.session.get("project1_feature_specs", [])
            scatter_specs = request.session.get("project1_scatter_specs", [])
            univariate_specs = request.session.get("project1_univariate_specs", [])
            highlight_key = None

            if not dataset:
                error = "Please upload a CSV file before creating a plot."
                if is_ajax_request(request):
                    return ajax_error(error)
            else:
                dataset_summary = get_dataset_summary(dataset)
                feature_options = dataset["features"]
                task_type = get_effective_task_type(dataset)
                model_options = training_model_options(task_type)
                result = calculate_average_target(dataset)
                training_result = request.session.get("project1_training_result")
                if training_result:
                    selected_model = training_result.get("model_value")
                    test_size_percent = training_result.get(
                        "test_size_percent", test_size_percent
                    )

                if action == "configure_dataset":
                    try:
                        missing_policy = request.POST.get("missing_policy", "impute_system")
                        
                        imputation_strategies = {}
                        if missing_policy == "impute_manual":
                            for key, val in request.POST.items():
                                if key.startswith("impute_strategy_"):
                                    feat_name = key[len("impute_strategy_"):]
                                    imputation_strategies[feat_name] = val
                        
                        dataset = configure_dataset(
                            dataset,
                            target=request.POST.get("target"),
                            features=request.POST.getlist("features"),
                            task_type=request.POST.get("task_type"),
                            drop_missing_rows=(missing_policy == "drop"),
                        )
                        dataset["missing_policy"] = missing_policy
                        dataset["imputation_strategies"] = imputation_strategies
                        
                        request.session["project1_dataset"] = dataset
                        reset_project1_outputs(request)
                        feature_specs = []
                        scatter_specs = []
                        univariate_specs = []
                        training_result = None
                        selected_model = None
                        result = calculate_average_target(dataset)
                        dataset_summary = get_dataset_summary(dataset)
                        feature_options = dataset["features"]
                        task_type = get_effective_task_type(dataset)
                        model_options = training_model_options(task_type)
                    except Exception as e:
                        error = f"Error updating dataset setup: {str(e)}"
                elif action == "feature_target":
                    selected_feature = request.POST.get("feature")
                    if selected_feature not in feature_options:
                        error = "Please select a valid feature column."
                    elif selected_feature in feature_specs:
                        highlight_key = feature_target_key(selected_feature)
                        duplicate_message = (
                            f"A plot for {selected_feature} already exists."
                        )
                    else:
                        if is_ajax_request(request):
                            try:
                                plot = save_feature_target_visualization(
                                    dataset, selected_feature
                                )
                                feature_specs = feature_specs + [selected_feature]
                                request.session["project1_feature_specs"] = (
                                    feature_specs
                                )
                                return JsonResponse(
                                    {
                                        "status": "created",
                                        "plot_type": "feature_target",
                                        "plot": plot_payload(plot),
                                    }
                                )
                            except Exception as e:
                                return ajax_error(f"Error creating plot: {str(e)}")
                        feature_specs = feature_specs + [selected_feature]
                        request.session["project1_feature_specs"] = feature_specs
                elif action == "scatter":
                    selected_x_feature = request.POST.get("x_feature")
                    selected_y_feature = request.POST.get("y_feature")

                    if (
                        selected_x_feature not in feature_options
                        or selected_y_feature not in feature_options
                    ):
                        error = "Please select two valid feature columns."
                    elif selected_x_feature == selected_y_feature:
                        error = "Please select two different feature columns."
                    else:
                        new_scatter = {
                            "x": selected_x_feature,
                            "y": selected_y_feature,
                        }
                        if new_scatter in scatter_specs:
                            highlight_key = scatter_key(
                                selected_x_feature, selected_y_feature
                            )
                            duplicate_message = (
                                f"A scatter plot for {selected_x_feature} vs "
                                f"{selected_y_feature} already exists."
                            )
                        else:
                            if is_ajax_request(request):
                                try:
                                    plot = save_scatter_visualization(
                                        dataset,
                                        selected_x_feature,
                                        selected_y_feature,
                                    )
                                    scatter_specs = scatter_specs + [new_scatter]
                                    request.session["project1_scatter_specs"] = (
                                        scatter_specs
                                    )
                                    return JsonResponse(
                                        {
                                            "status": "created",
                                            "plot_type": "scatter",
                                            "plot": plot_payload(plot),
                                        }
                                    )
                                except Exception as e:
                                    return ajax_error(f"Error creating plot: {str(e)}")
                            scatter_specs = scatter_specs + [new_scatter]
                            request.session["project1_scatter_specs"] = scatter_specs
                elif action == "remove_feature_target":
                    selected_feature = request.POST.get("feature")
                    if selected_feature not in feature_specs:
                        error = "That feature plot is not currently shown."
                    else:
                        feature_specs = [
                            feature
                            for feature in feature_specs
                            if feature != selected_feature
                        ]
                        request.session["project1_feature_specs"] = feature_specs
                        if is_ajax_request(request):
                            return JsonResponse(
                                {
                                    "status": "removed",
                                    "plot_type": "feature_target",
                                    "key": feature_target_key(selected_feature),
                                }
                            )
                elif action == "remove_scatter":
                    selected_x_feature = request.POST.get("x_feature")
                    selected_y_feature = request.POST.get("y_feature")
                    selected_scatter = {
                        "x": selected_x_feature,
                        "y": selected_y_feature,
                    }
                    if selected_scatter not in scatter_specs:
                        error = "That scatter plot is not currently shown."
                    else:
                        scatter_specs = [
                            spec for spec in scatter_specs if spec != selected_scatter
                        ]
                        request.session["project1_scatter_specs"] = scatter_specs
                        if is_ajax_request(request):
                            return JsonResponse(
                                {
                                    "status": "removed",
                                    "plot_type": "scatter",
                                    "key": scatter_key(
                                        selected_x_feature, selected_y_feature
                                    ),
                                }
                            )
                elif action == "univariate":
                    selected_feature = request.POST.get("feature")
                    if selected_feature not in dataset["column_names"]:
                        error = "Please select a valid column."
                    elif selected_feature in univariate_specs:
                        highlight_key = univariate_key(selected_feature)
                        duplicate_message = (
                            f"A distribution plot for {selected_feature} already exists."
                        )
                    else:
                        if is_ajax_request(request):
                            try:
                                plot = save_univariate_visualization(
                                    dataset, selected_feature
                                )
                                univariate_specs = univariate_specs + [selected_feature]
                                request.session["project1_univariate_specs"] = (
                                    univariate_specs
                                )
                                return JsonResponse(
                                    {
                                        "status": "created",
                                        "plot_type": "univariate",
                                        "plot": plot_payload(plot),
                                    }
                                )
                            except Exception as e:
                                return ajax_error(f"Error creating plot: {str(e)}")
                        univariate_specs = univariate_specs + [selected_feature]
                        request.session["project1_univariate_specs"] = univariate_specs
                elif action == "remove_univariate":
                    selected_feature = request.POST.get("feature")
                    if selected_feature not in univariate_specs:
                        error = "That distribution plot is not currently shown."
                    else:
                        univariate_specs = [
                            f for f in univariate_specs if f != selected_feature
                        ]
                        request.session["project1_univariate_specs"] = univariate_specs
                        if is_ajax_request(request):
                            return JsonResponse(
                                {
                                    "status": "removed",
                                    "plot_type": "univariate",
                                    "key": univariate_key(selected_feature),
                                }
                            )
                elif action == "train_model":
                    selected_model = request.POST.get("model")
                    test_size_percent = request.POST.get("test_size", "20")
                    selected_params = request.POST.getlist("hyperparameters")
                    enable_poly = request.POST.get("enable_poly") == "true" or request.POST.get("enable_poly") == "on"
                    try:
                        training_result = train_model(
                            dataset,
                            selected_model,
                            test_size_percent,
                            selected_params=selected_params,
                            enable_poly=enable_poly,
                        )
                        request.session["project1_training_result"] = training_result
                        selected_model = training_result["model_value"]
                        test_size_percent = training_result["test_size_percent"]
                    except Exception as e:
                        error = f"Error training model: {str(e)}"
                elif action == "compare_all":
                    test_size_percent = request.POST.get("test_size", "20")
                    try:
                        results = train_all_and_compare(dataset, test_size_percent)
                        request.session["project1_comparison_results"] = results
                    except Exception as e:
                        error = f"Error running comparison: {str(e)}"
                else:
                    error = "Unknown action."

                if is_ajax_request(request):
                    if duplicate_message:
                        return JsonResponse(
                            {
                                "status": "duplicate",
                                "message": duplicate_message,
                                "key": highlight_key,
                            }
                        )
                    if error:
                        return ajax_error(error)

                try:
                    overview_visualizations = save_overview_visualizations(dataset)
                    feature_target_plots, scatter_plots, univariate_plots = build_saved_visualizations(
                        dataset,
                        feature_specs,
                        scatter_specs,
                        univariate_specs,
                        highlight_key=highlight_key,
                    )
                except Exception as e:
                    error = f"Error creating plot: {str(e)}"
                
                if not is_ajax_request(request):
                    return redirect("project1:index")
    else:
        dataset = ensure_dataset_configuration(request.session.get("project1_dataset"))
        if dataset:
            request.session["project1_dataset"] = dataset
            dataset_summary = get_dataset_summary(dataset)
            feature_options = dataset["features"]
            task_type = get_effective_task_type(dataset)
            model_options = training_model_options(task_type)
            result = calculate_average_target(dataset)
            training_result = request.session.get("project1_training_result")
            if training_result:
                selected_model = training_result.get("model_value")
                test_size_percent = training_result.get(
                    "test_size_percent", test_size_percent
                )
            feature_specs = request.session.get("project1_feature_specs", [])
            scatter_specs = request.session.get("project1_scatter_specs", [])
            univariate_specs = request.session.get("project1_univariate_specs", [])
            overview_visualizations = save_overview_visualizations(dataset)
            feature_target_plots, scatter_plots, univariate_plots = build_saved_visualizations(
                dataset, feature_specs, scatter_specs, univariate_specs
            )

    selected_feature, selected_x_feature, selected_y_feature = (
        selected_features_from_options(
            feature_options,
            selected_feature,
            selected_x_feature,
            selected_y_feature,
        )
    )
    assumption_panel = None
    column_config_options = []
    id_columns = []
    if dataset_summary and dataset:
        assumption_panel = build_dataset_assumptions(
            dataset, task_type, test_size_percent
        )
        column_config_options = build_column_config_options(dataset)
        id_columns = [col["name"] for col in column_config_options if col["likely_id"] and not col["is_target"]]

    content_menu_items = [
        {
            "label": "1. DATA MANAGEMENT",
            "href": "#data-management",
            "children": [
                {"label": "Dataset Examples", "href": "#dataset-examples"},
                {"label": "Upload Custom CSV", "href": "#upload-csv"},
                {"label": "Your Datasets", "href": "#your-datasets"},
            ],
        },
        {
            "label": "2. DATASET AUDIT",
            "href": "#dataset-audit",
            "children": [
                {"label": "Column Statistics", "href": "#column-statistics"},
                {"label": "Missing-value handling", "href": "#missing-value-handling"},
                {"label": "Feature Distributions", "href": "#feature-distributions"},
                {"label": "Feature Correlation Matrix", "href": "#feature-correlation-matrix"},
                {"label": "2D Scatter Plots", "href": "#2d-scatter-plots"},
                {"label": "Feature columns used for training", "href": "#feature-columns-for-training"},
            ],
        },
        {
            "label": "3. MODEL TRAINING",
            "href": "#model-training",
            "children": [
                {"label": "Manual Explorer", "href": "#manual-explorer"},
                {"label": "Train All & Compare", "href": "#train-all-and-compare"},
            ],
        }
    ]

    return render(
        request,
        "project1/index.html",
        {
            "result": result,
            "error": error,
            "upload_error": upload_error,
            "duplicate_message": duplicate_message,
            "overview_visualizations": overview_visualizations,
            "feature_target_plots": feature_target_plots,
            "scatter_plots": scatter_plots,
            "dataset_summary": dataset_summary,
            "feature_options": feature_options,
            "selected_feature": selected_feature,
            "selected_x_feature": selected_x_feature,
            "selected_y_feature": selected_y_feature,
            "task_type": task_type,
            "model_options": model_options,
            "selected_model": selected_model
            or (model_options[0]["value"] if model_options else None),
            "test_size_percent": test_size_percent,
            "training_result": training_result,
            "assumption_panel": assumption_panel,
            "column_config_options": column_config_options,
            "task_options": TASK_OPTIONS,
            "content_menu_items": content_menu_items,
            "uploaded_datasets": request.session.get("project1_uploaded_datasets", []),
            "active_dataset_name": dataset.get("name") if dataset else None,
            "audit_columns": build_audit_columns(dataset) if dataset else None,
            "drop_missing_info": get_missing_rows_info(dataset) if dataset else None,
            "has_missing_values": any(col["missing_count"] > 0 for col in build_audit_columns(dataset)) if dataset else False,
            "correlation_data": build_correlation_data(dataset) if dataset else None,
            "id_columns": id_columns,
            "univariate_plots": univariate_plots if dataset else [],
            "comparison_results": request.session.get("project1_comparison_results", []),
        },
    )
