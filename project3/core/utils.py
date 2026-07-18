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
    per_class = max(1, max_rows // len(CLASS_IDS))
    sampled = []
    for class_id in CLASS_IDS:
        sampled.extend(by_class[class_id][:per_class])
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
    
    max_val = max(values) if values else 100
    axis.set_ylim(0, max_val * 1.15 if max_val > 0 else 10)
    
    # Rotate labels
    plt.xticks(rotation=15, ha='right')
    
    axis.grid(axis="y", linestyle='--', alpha=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        if yval is not None:
            if isinstance(yval, float):
                label_text = f"{yval:.2f}" if yval % 1 != 0 else f"{int(yval)}"
            else:
                label_text = str(yval)
            axis.text(
                bar.get_x() + bar.get_width()/2.0,
                yval + (max_val * 0.015),
                label_text,
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                color='#002D3A'
            )
            
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return settings.MEDIA_URL + f"project3/{filename}"
