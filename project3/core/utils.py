import os
import csv
from functools import lru_cache
from urllib.error import URLError
from urllib.request import urlretrieve
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from django.conf import settings

CLASS_NAMES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
CLASS_IDS = list(CLASS_NAMES.keys())

TRAIN_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
TEST_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
DEFAULT_TRAIN_SIZE = 8000
DEFAULT_TEST_SIZE = 2000

FALLBACK_ROWS = [
    (1, "UN leaders discuss humanitarian aid", "Diplomats met to coordinate relief and peace talks."),
    (1, "Election observers monitor vote", "International observers watched polling stations after unrest."),
    (1, "Trade ministers reach border agreement", "Neighboring governments announced a new customs accord."),
    (2, "Local club wins final", "The team scored twice late to win the championship."),
    (2, "Tennis star reaches semifinal", "A strong serve helped the player advance in straight sets."),
    (2, "Coach praises defensive effort", "The squad held its rival scoreless during the match."),
    (3, "Stocks rise after earnings reports", "Technology shares lifted the market after quarterly results."),
    (3, "Oil prices affect airline profits", "Fuel costs pressured carriers despite strong travel demand."),
    (3, "Central bank keeps rates steady", "Investors watched the decision for signs of inflation risk."),
    (4, "New chip improves battery life", "Researchers introduced a processor for efficient mobile devices."),
    (4, "Space telescope sends images", "Scientists released detailed observations of distant galaxies."),
    (4, "Software update fixes security issue", "The patch closes a vulnerability in network services."),
]

def parse_sample_size(value, default):
    if value == "all":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default

def parse_float(value, default, minimum=0.0, maximum=1.0):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)

def cache_dir():
    path = os.path.join(settings.MEDIA_ROOT, "project3", "ag_news")
    os.makedirs(path, exist_ok=True)
    return path

def artifact_dir():
    path = os.path.join(settings.MEDIA_ROOT, "project3")
    os.makedirs(path, exist_ok=True)
    return path

def cached_csv_path(split):
    return os.path.join(cache_dir(), f"{split}.csv")

def ensure_ag_news_file(split):
    path = cached_csv_path(split)
    if os.path.exists(path):
        return path
    url = TRAIN_URL if split == "train" else TEST_URL
    urlretrieve(url, path)
    return path

def read_ag_news_csv(path):
    examples = []
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if len(row) < 3:
                continue
            label = int(row[0])
            text = f"{row[1]} {row[2]}".strip()
            examples.append({"label": label, "text": text})
    return examples

def balanced_sample(examples, max_rows):
    if max_rows is None or len(examples) <= max_rows:
        return list(examples)
    by_class = {class_id: [] for class_id in CLASS_IDS}
    for example in examples:
        by_class[example["label"]].append(example)
    sampled = []
    base_per_class = max_rows // len(CLASS_IDS)
    remainder = max_rows % len(CLASS_IDS)
    for index, class_id in enumerate(CLASS_IDS):
        # Keep the sample balanced while preserving the exact requested row count.
        class_limit = base_per_class + (1 if index < remainder else 0)
        sampled.extend(by_class[class_id][:class_limit])
    return sampled[:max_rows]

def build_fallback_dataset():
    examples = [{"label": label, "text": f"{title}. {description}"} for label, title, description in FALLBACK_ROWS]
    return examples * 8, examples * 3

@lru_cache(maxsize=8)
def load_ag_news_dataset(train_size, test_size):
    try:
        train_examples = read_ag_news_csv(ensure_ag_news_file("train"))
        test_examples = read_ag_news_csv(ensure_ag_news_file("test"))
        source = "AG News CSV cache"
    except (OSError, URLError, ValueError):
        train_examples, test_examples = build_fallback_dataset()
        source = "built-in fallback sample"
    return {
        "train": balanced_sample(train_examples, train_size),
        "test": balanced_sample(test_examples, test_size),
        "full_train_rows": len(train_examples),
        "full_test_rows": len(test_examples),
        "source": source,
    }

def save_bar_plot(filename, title, labels, values, ylabel, color="#00a6b2"):
    path = os.path.join(artifact_dir(), filename)
    figure, axis = plt.subplots(figsize=(8.5, 5))
    bars = axis.bar(labels, values, color=color)
    axis.set_title(title, pad=15, fontweight='bold', color='#002D3A')
    axis.set_ylabel(ylabel, fontweight='bold')
    
    min_val = min(values) if values else 0
    max_val = max(values) if values else 100
    span = max_val - min_val
    padding = span * 0.15 if span > 0 else 10
    
    lower_lim = min_val - padding if min_val < 0 else 0
    upper_lim = max_val + padding if max_val > 0 else 10
    axis.set_ylim(lower_lim, upper_lim)
    
    # Rotate labels
    plt.xticks(rotation=15, ha='right')
    
    axis.grid(axis="y", linestyle='--', alpha=0.5)
    axis.axhline(0, color='#333333', linewidth=1.2, zorder=2)
    
    for bar in bars:
        yval = bar.get_height()
        if yval is not None:
            if isinstance(yval, float):
                label_text = f"{yval:.2f}" if yval % 1 != 0 else f"{int(yval)}"
            else:
                label_text = str(yval)
            offset = (span * 0.015) if yval >= 0 else -(span * 0.03)
            va_align = 'bottom' if yval >= 0 else 'top'
            axis.text(
                bar.get_x() + bar.get_width()/2.0,
                yval + offset,
                label_text,
                ha='center',
                va=va_align,
                fontsize=9,
                fontweight='bold',
                color='#002D3A'
            )
            
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return settings.MEDIA_URL + f"project3/{filename}"


def save_active_learning_scatter_plot(train_examples, active_learning, filename="active_learning_scatter.png"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    import numpy as np

    train_texts = [ex["text"] for ex in train_examples]
    train_labels = [ex["label"] for ex in train_examples]
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train = vectorizer.fit_transform(train_texts)
    
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_train)
    
    path = os.path.join(artifact_dir(), filename)
    figure, axis = plt.subplots(figsize=(7, 5))
    
    # Plot all background points
    axis.scatter(
        X_2d[:, 0], 
        X_2d[:, 1], 
        c="#cbd5e1", 
        alpha=0.4, 
        s=6, 
        label="Dataset background",
        edgecolors="none"
    )
    
    # Plot selected query points
    selected_indices = active_learning.get("selected_indices", [])
    if len(selected_indices) > 0:
        selected_indices = np.array(selected_indices)
        X_selected = X_2d[selected_indices]
        labels_selected = np.array(train_labels)[selected_indices]
        
        # Color palette for classes: World, Sports, Business, Sci/Tech
        # Colors: World: blue, Sports: green, Business: orange, Sci/Tech: purple
        class_colors = {1: "#3b82f6", 2: "#10b981", 3: "#f97316", 4: "#8b5cf6"}
        
        for class_id in CLASS_IDS:
            class_mask = labels_selected == class_id
            if np.any(class_mask):
                axis.scatter(
                    X_selected[class_mask, 0],
                    X_selected[class_mask, 1],
                    c=class_colors[class_id],
                    s=40,
                    marker="*",
                    label=CLASS_NAMES[class_id],
                    edgecolors="none"
                )

    axis.set_title("Active Learning Queries in Semantic Space", pad=15, fontweight='bold', color='#002D3A')
    axis.legend(loc="upper right", frameon=True, fontsize=9)
    axis.set_xlabel("SVD Component 1", fontweight='bold')
    axis.set_ylabel("SVD Component 2", fontweight='bold')
    
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    
    return settings.MEDIA_URL + f"project3/{filename}"
