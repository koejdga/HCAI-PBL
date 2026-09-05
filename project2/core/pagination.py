from urllib.parse import urlencode
from django.core.paginator import Paginator

from .constants import (
    CLASS_NAMES,
    DATASET_PAGE_SIZE,
    DISPLAY_COLUMN_LABELS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TREE_PAGE_SIZE,
)
from .dataset import format_display_value


def build_dataset_page(penguins, request):
    selected_row_id = request.GET.get("selected-row")
    page_number = request.GET.get("dataset-page", 1)
    base_query = request.GET.copy()
    base_query.pop("dataset-page", None)
    base_query.pop("format", None)

    table_rows = []
    selected_row = None

    for row_id, row in penguins.iterrows():
        values = {column: row[column] for column in FEATURE_COLUMNS + [TARGET_COLUMN]}
        display_values = {
            column: format_display_value(column, value)
            for column, value in values.items()
        }
        row_data = {
            "id": str(row_id),
            "values": values,
            "display_values": display_values,
            "selected": str(row_id) == selected_row_id,
        }
        table_rows.append(row_data)

        if row_data["selected"]:
            selected_row = row_data

    if selected_row is None and table_rows:
        selected_row = table_rows[0]
        selected_row["selected"] = True
        selected_row_id = selected_row["id"]

    base_query["selected-row"] = selected_row_id
    paginator = Paginator(table_rows, DATASET_PAGE_SIZE)
    page_obj = paginator.get_page(page_number)

    def page_url(number):
        query = base_query.copy()
        query["dataset-page"] = number
        return f"?{urlencode(query, doseq=True)}#dataset-table-container"

    page_range = paginator.get_elided_page_range(
        page_obj.number,
        on_each_side=2,
        on_ends=1,
    )

    page_links = [
        {
            "number": number,
            "url": page_url(number) if isinstance(number, int) else None,
            "is_current": number == page_obj.number,
            "is_ellipsis": not isinstance(number, int),
        }
        for number in page_range
    ]

    return {
        "dataset_columns": FEATURE_COLUMNS + [TARGET_COLUMN],
        "dataset_column_labels": [
            DISPLAY_COLUMN_LABELS[column]
            for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        ],
        "dataset_page": page_obj,
        "dataset_page_links": page_links,
        "dataset_previous_url": page_url(page_obj.previous_page_number())
        if page_obj.has_previous()
        else None,
        "dataset_next_url": page_url(page_obj.next_page_number())
        if page_obj.has_next()
        else None,
        "selected_dataset_row": selected_row,
        "selected_row_id": selected_row_id,
        "selected_dataset_row_display_values": [
            {
                "label": DISPLAY_COLUMN_LABELS[column],
                "value": selected_row["display_values"][column],
            }
            for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        ] if selected_row is not None else [],
    }


def build_tree_datapoint_page(penguins, request):
    """Separate, compact pagination for exploring how the tree evaluates individual datapoints."""
    trace_row_id = request.GET.get("trace-row")
    page_number = request.GET.get("tree-page", 1)
    base_query = request.GET.copy()
    base_query.pop("tree-page", None)
    base_query.pop("format", None)

    species_groups = {c: [] for c in CLASS_NAMES}
    traced_row = None

    for row_id, row in penguins.iterrows():
        values = {column: row[column] for column in FEATURE_COLUMNS + [TARGET_COLUMN]}
        display_values = {
            column: format_display_value(column, value)
            for column, value in values.items()
        }
        is_traced = str(row_id) == trace_row_id
        row_data = {
            "id": str(row_id),
            "values": values,
            "display_values": display_values,
            "is_traced": is_traced,
        }
        target_val = row[TARGET_COLUMN]
        if target_val in species_groups:
            species_groups[target_val].append(row_data)
        else:
            species_groups.setdefault(target_val, []).append(row_data)

        if is_traced:
            traced_row = row_data

    # Interleave species so each page alternates Adelie, Chinstrap, and Gentoo,
    # ensuring that clicking consecutive rows traces visibly different branches through the tree.
    table_rows = []
    max_len = max(len(grp) for grp in species_groups.values()) if species_groups else 0
    class_order = [c for c in CLASS_NAMES if c in species_groups] + [
        k for k in species_groups if k not in CLASS_NAMES
    ]
    for i in range(max_len):
        for c in class_order:
            if i < len(species_groups[c]):
                table_rows.append(species_groups[c][i])

    paginator = Paginator(table_rows, TREE_PAGE_SIZE)
    page_obj = paginator.get_page(page_number)

    def tree_page_url(number):
        query = base_query.copy()
        query["tree-page"] = number
        return f"?{urlencode(query, doseq=True)}#tree-datapoint-explorer"

    page_range = paginator.get_elided_page_range(
        page_obj.number,
        on_each_side=1,
        on_ends=1,
    )

    page_links = [
        {
            "number": number,
            "url": tree_page_url(number) if isinstance(number, int) else None,
            "is_current": number == page_obj.number,
            "is_ellipsis": not isinstance(number, int),
        }
        for number in page_range
    ]

    tree_page_param = request.GET.get("tree-page")
    tree_explorer_open = bool(
        trace_row_id or (tree_page_param is not None and str(tree_page_param).strip() != "")
    )

    return {
        "tree_page": page_obj,
        "tree_page_links": page_links,
        "tree_previous_url": tree_page_url(page_obj.previous_page_number())
        if page_obj.has_previous()
        else None,
        "tree_next_url": tree_page_url(page_obj.next_page_number())
        if page_obj.has_next()
        else None,
        "traced_row": traced_row,
        "traced_row_id": trace_row_id,
        "tree_explorer_open": tree_explorer_open,
    }
