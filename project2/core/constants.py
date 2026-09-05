"""Constants and configuration settings for Project 2."""

TARGET_COLUMN = "species"
CATEGORICAL_FEATURES = ["island", "sex"]
NUMERIC_FEATURES = [
    "year",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
FEATURE_EFFECT_NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

DISPLAY_COLUMN_LABELS = {
    "island": "Island",
    "sex": "Sex",
    "year": "Year",
    "bill_length_mm": "Bill Length",
    "bill_depth_mm": "Bill Depth",
    "flipper_length_mm": "Flipper Length",
    "body_mass_g": "Body Mass",
    "species": "Species",
}

# Candidate sizes used to compare simple and more detailed trees.
MAX_LEAF_OPTIONS = [2, 3, 4, 5, 6, 8, 10, 12, 15]
C_OPTIONS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]

DEFAULT_LAMBDA = 0.20
MIN_LAMBDA = 0.0
MAX_LAMBDA = 1.0
DEFAULT_THETA = 0.03

COUNTERFACTUAL_ATTEMPTS = [
    {"N": 2000, "variance_scale": 0.10},
    {"N": 4000, "variance_scale": 0.20},
    {"N": 8000, "variance_scale": 0.35},
]
DATASET_PAGE_SIZE = 12
TREE_PAGE_SIZE = 6
CLASS_NAMES = ["Adelie", "Chinstrap", "Gentoo"]
CONTENT_MENU_ITEMS = [
    {"label": "Data transparency", "href": "#data-transparency"},
    {"label": "Model selection", "href": "#model-selection"},
    {"label": "Results", "href": "#model-results"},
    {"label": "Tree explanation", "href": "#tree-explanation"},
    {"label": "Counterfactuals", "href": "#counterfactuals"},
    {"label": "Feature effects", "href": "#feature-effects"},
]
PROJECT2_CACHE_TIMEOUT = 60 * 30
