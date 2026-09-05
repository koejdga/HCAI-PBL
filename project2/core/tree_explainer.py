"""Decision tree rule extraction, individual prediction path tracing, and interactive SVG visualization."""

import numpy as np
import pandas as pd

from .constants import TARGET_COLUMN
from .dataset import format_feature_name


def extract_tree_rules(pipeline):
    """Extract all root-to-leaf decision paths into human-readable rule lists (Lecture 4: CORELS style)."""
    tree = pipeline.named_steps["model"].tree_
    preprocessor = pipeline.named_steps["preprocess"]
    classes = list(pipeline.named_steps["model"].classes_)
    feature_names = [format_feature_name(n) for n in preprocessor.get_feature_names_out()]

    rules = []

    def traverse(node_id, current_conditions):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        if left == -1 and right == -1:
            val_counts = tree.value[node_id][0]
            total_val = sum(val_counts)
            dom_idx = int(np.argmax(val_counts))
            dom_class = classes[dom_idx]
            purity = (val_counts[dom_idx] / total_val * 100) if total_val > 0 else 0
            rules.append({
                "rule_id": len(rules) + 1,
                "node_id": node_id,
                "conditions": list(current_conditions),
                "predicted_species": dom_class,
                "purity_percent": round(purity, 1),
                "samples": int(tree.n_node_samples[node_id]),
            })
            return

        feat_idx = tree.feature[node_id]
        feat_name = feature_names[feat_idx]
        thresh = tree.threshold[node_id]

        if " = " in feat_name:
            left_cond = f"{feat_name} is False"
            right_cond = f"{feat_name} is True"
        else:
            unit = " mm" if "Length" in feat_name or "Depth" in feat_name else (" g" if "Mass" in feat_name else "")
            left_cond = f"{feat_name} ≤ {thresh:.1f}{unit}"
            right_cond = f"{feat_name} > {thresh:.1f}{unit}"

        if left != -1:
            traverse(left, current_conditions + [left_cond])
        if right != -1:
            traverse(right, current_conditions + [right_cond])

    traverse(0, [])
    return rules


def trace_penguin_path(pipeline, penguin_row):
    """Trace the exact decision sequence followed by a single penguin row from root to leaf."""
    tree = pipeline.named_steps["model"].tree_
    preprocessor = pipeline.named_steps["preprocess"]
    classes = list(pipeline.named_steps["model"].classes_)
    feature_names = [format_feature_name(n) for n in preprocessor.get_feature_names_out()]

    row_df = pd.DataFrame([penguin_row])
    X_trans = preprocessor.transform(row_df)

    node_id = 0
    visited_nodes = [node_id]
    visited_edges = []
    steps = []

    while tree.children_left[node_id] != -1:
        feat_idx = tree.feature[node_id]
        feat_name = feature_names[feat_idx]
        thresh = tree.threshold[node_id]
        val = X_trans[0, feat_idx]
        cond = val <= thresh
        
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        next_node = left if cond else right
        
        visited_edges.append((node_id, int(next_node)))
        
        if " = " in feat_name:
            actual_str = "True" if val > 0.5 else "False"
            decision_str = f"{feat_name}: {actual_str} ➔ {'Left (False)' if cond else 'Right (True)'}"
        else:
            unit = " mm" if "Length" in feat_name or "Depth" in feat_name else (" g" if "Mass" in feat_name else "")
            decision_str = f"{feat_name} ({val:.1f}{unit}) {'≤' if cond else '>'} {thresh:.1f}{unit} ➔ {'Branch Left (True)' if cond else 'Branch Right (False)'}"

        steps.append({
            "step_num": len(steps) + 1,
            "node_id": node_id,
            "feature": feat_name,
            "value": f"{val:.1f}",
            "operator": "≤" if cond else ">",
            "threshold": f"{thresh:.1f}",
            "branch": "Left" if cond else "Right",
            "description": decision_str,
        })
        
        node_id = int(next_node)
        visited_nodes.append(node_id)

    # Leaf reached
    val_counts = tree.value[node_id][0]
    total_val = sum(val_counts)
    dom_idx = int(np.argmax(val_counts))
    dom_class = classes[dom_idx]
    purity = (val_counts[dom_idx] / total_val * 100) if total_val > 0 else 0

    return {
        "visited_nodes": visited_nodes,
        "visited_edges": visited_edges,
        "steps": steps,
        "terminal_node_id": node_id,
        "predicted_species": dom_class,
        "actual_species": penguin_row.get(TARGET_COLUMN, "Unknown"),
        "confidence_percent": round(purity, 1),
        "samples": int(tree.n_node_samples[node_id]),
    }


def generate_tree_svg(pipeline, highlighted_nodes=None, highlighted_edges=None):
    """Generate interactive, resolution-independent SVG with hover tooltips and path highlighting."""
    tree = pipeline.named_steps["model"].tree_
    preprocessor = pipeline.named_steps["preprocess"]
    classes = list(pipeline.named_steps["model"].classes_)
    feature_names = [format_feature_name(n) for n in preprocessor.get_feature_names_out()]

    if highlighted_nodes is None:
        highlighted_nodes = set()
    else:
        highlighted_nodes = set(highlighted_nodes)
    if highlighted_edges is None:
        highlighted_edges = set()
    else:
        highlighted_edges = set(highlighted_edges)

    node_depth = {}
    leaves = []

    def compute_positions(node_id, depth=0):
        node_depth[node_id] = depth
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left == -1 and right == -1:
            leaves.append(node_id)
        else:
            if left != -1:
                compute_positions(left, depth + 1)
            if right != -1:
                compute_positions(right, depth + 1)

    compute_positions(0, 0)

    node_coords = {}
    leaf_spacing = 200
    node_w = 170
    node_h = 75
    y_spacing = 110
    margin_x = 40
    margin_y = 40

    for i, leaf_id in enumerate(leaves):
        node_coords[leaf_id] = (
            margin_x + i * leaf_spacing + node_w / 2,
            margin_y + node_depth[leaf_id] * y_spacing + node_h / 2,
        )

    def assign_internal_x(node_id):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left == -1 and right == -1:
            return node_coords[node_id][0]
        left_x = assign_internal_x(left)
        right_x = assign_internal_x(right)
        mid_x = (left_x + right_x) / 2
        y = margin_y + node_depth[node_id] * y_spacing + node_h / 2
        node_coords[node_id] = (mid_x, y)
        return mid_x

    assign_internal_x(0)

    max_x = max(coords[0] for coords in node_coords.values()) + node_w / 2 + margin_x
    max_y = max(coords[1] for coords in node_coords.values()) + node_h / 2 + margin_y

    svg_parts = [
        f'<svg class="interactive-tree-svg" viewBox="0 0 {int(max_x)} {int(max_y)}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">',
        """<defs>
            <filter id="active-glow" x="-30%" y="-30%" width="160%" height="160%">
                <feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="#00c2d3" flood-opacity="0.8"/>
            </filter>
        </defs>""",
        '<g class="tree-edges">',
    ]

    # Draw edges
    for node_id in range(tree.node_count):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        px, py = node_coords[node_id]
        p_bottom = (px, py + node_h / 2)

        for child, is_left in [(left, True), (right, False)]:
            if child != -1:
                cx, cy = node_coords[child]
                c_top = (cx, cy - node_h / 2)
                is_active_edge = (node_id, child) in highlighted_edges
                edge_cls = "tree-edge active-edge" if is_active_edge else "tree-edge"
                edge_stroke = "#00c2d3" if is_active_edge else "#cbd5e1"
                edge_width = "4" if is_active_edge else "2"

                mid_y = (p_bottom[1] + c_top[1]) / 2
                path_d = f"M {p_bottom[0]} {p_bottom[1]} C {p_bottom[0]} {mid_y}, {c_top[0]} {mid_y}, {c_top[0]} {c_top[1]}"
                svg_parts.append(
                    f'<path d="{path_d}" fill="none" stroke="{edge_stroke}" stroke-width="{edge_width}" class="{edge_cls}" />'
                )

                label_text = "True (≤)" if is_left else "False (>)"
                lx = p_bottom[0] * 0.65 + c_top[0] * 0.35
                ly = mid_y - 4
                label_color = "#00838f" if is_active_edge else "#64748b"
                svg_parts.append(
                    f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="11" font-weight="600" fill="{label_color}" text-anchor="middle">{label_text}</text>'
                )

    svg_parts.append('</g><g class="tree-nodes">')

    # Draw nodes
    for node_id in range(tree.node_count):
        x, y = node_coords[node_id]
        rx = x - node_w / 2
        ry = y - node_h / 2

        is_leaf = tree.children_left[node_id] == -1 and tree.children_right[node_id] == -1
        gini = tree.impurity[node_id]
        samples = tree.n_node_samples[node_id]
        val_counts = tree.value[node_id][0]
        total_val = sum(val_counts)
        val_pcts = [(v / total_val) if total_val > 0 else 0 for v in val_counts]
        class_counts = [
            int(round(v * samples / total_val)) if total_val > 0 and total_val <= 1.5 else int(round(v))
            for v in val_counts
        ]
        dominant_idx = int(np.argmax(val_counts))
        dominant_class = classes[dominant_idx]

        is_active = node_id in highlighted_nodes

        class_colors = {
            "Adelie": {"bg": "#fff7ed", "border": "#ea580c"},
            "Chinstrap": {"bg": "#f0fdf4", "border": "#16a34a"},
            "Gentoo": {"bg": "#eff6ff", "border": "#2563eb"},
        }
        style_info = class_colors.get(
            dominant_class, {"bg": "#f8fafc", "border": "#64748b"}
        )

        node_border = "#00c2d3" if is_active else style_info["border"]
        stroke_w = "3.5" if is_active else "1.5"
        filter_attr = 'filter="url(#active-glow)"' if is_active else ""
        node_cls = "tree-node active-node" if is_active else "tree-node"

        data_attrs = (
            f'data-node-id="{node_id}" '
            f'data-is-leaf="{"true" if is_leaf else "false"}" '
            f'data-gini="{gini:.3f}" '
            f'data-samples="{samples}" '
            f'data-dominant="{dominant_class}" '
            f'data-c0-name="{classes[0]}" data-c0-count="{class_counts[0]}" data-c0-pct="{val_pcts[0]*100:.1f}%" '
            f'data-c1-name="{classes[1]}" data-c1-count="{class_counts[1]}" data-c1-pct="{val_pcts[1]*100:.1f}%" '
            f'data-c2-name="{classes[2]}" data-c2-count="{class_counts[2]}" data-c2-pct="{val_pcts[2]*100:.1f}%" '
        )

        svg_parts.append(f'<g class="{node_cls}" {data_attrs} style="cursor: pointer;">')
        svg_parts.append(
            f'<rect x="{rx}" y="{ry}" width="{node_w}" height="{node_h}" rx="8" ry="8" fill="{style_info["bg"]}" stroke="{node_border}" stroke-width="{stroke_w}" {filter_attr} />'
        )

        if not is_leaf:
            feat_idx = tree.feature[node_id]
            feat_name = feature_names[feat_idx]
            thresh = tree.threshold[node_id]

            if " = " in feat_name:
                cond_text = f"{feat_name}"
                split_display = "is True"
            else:
                unit = " mm" if "Length" in feat_name or "Depth" in feat_name else (" g" if "Mass" in feat_name else "")
                cond_text = f"{feat_name}"
                split_display = f"≤ {thresh:.1f}{unit}"

            svg_parts.append(
                f'<text x="{x}" y="{ry + 22}" font-size="12" font-weight="700" fill="#0f172a" text-anchor="middle">{cond_text}</text>'
            )
            svg_parts.append(
                f'<text x="{x}" y="{ry + 40}" font-size="12" font-weight="600" fill="#00838f" text-anchor="middle">{split_display}</text>'
            )
            svg_parts.append(
                f'<text x="{x}" y="{ry + 60}" font-size="10" fill="#64748b" text-anchor="middle">Gini: {gini:.2f} | Samples: {samples}</text>'
            )
        else:
            purity = val_pcts[dominant_idx] * 100
            svg_parts.append(
                f'<text x="{x}" y="{ry + 26}" font-size="13" font-weight="800" fill="{style_info["border"]}" text-anchor="middle">★ {dominant_class}</text>'
            )
            svg_parts.append(
                f'<text x="{x}" y="{ry + 46}" font-size="11" font-weight="600" fill="#334155" text-anchor="middle">Purity: {purity:.1f}%</text>'
            )
            svg_parts.append(
                f'<text x="{x}" y="{ry + 62}" font-size="10" fill="#64748b" text-anchor="middle">Samples: {samples}</text>'
            )

        svg_parts.append("</g>")

    svg_parts.append("</g></svg>")
    return "\n".join(svg_parts)
