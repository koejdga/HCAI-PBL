# project3/core/true_l2d.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from .utils import CLASS_IDS, CLASS_NAMES
from .deferral import build_confusion_rows, simulated_expert_predict


class CSSLoss_2(nn.Module):
    def __init__(self, query_cost=0.0):
        super(CSSLoss_2, self).__init__()
        self.query_cost = query_cost

    def forward(self, logits, y_true, m_expert):
        batch_size, num_classes = logits.shape
        K = num_classes - 1  # 4 standard classes [cite: 218]

        log_probs = torch.log_softmax(logits, dim=-1)

        # 1. Track whether the expert is correct (1.0) or wrong (0.0)
        expert_correct = (y_true == m_expert).float().unsqueeze(1)

        # 2. Get the true class one-hot representation
        y_one_hot = torch.zeros(batch_size, K, device=logits.device)
        y_zero_indexed = (y_true - 1).long()
        y_one_hot.scatter_(1, y_zero_indexed.unsqueeze(1), 1.0)

        # 3. Calculate mathematically simplified Cost-Sensitive Multipliers:
        # multiplier_normal = alpha * (1 - expert_correct) + y_one_hot
        multiplier_normal = self.query_cost * (1.0 - expert_correct) + y_one_hot
        
        # multiplier_defer = expert_correct * (1 - alpha)
        multiplier_defer = expert_correct * (1.0 - self.query_cost)

        # Combine normal and deferral multipliers for all K+1 options
        multiplier = torch.cat([multiplier_normal, multiplier_defer], dim=1)

        # 4. Calculate expected loss [cite: 262, 371]
        loss = -torch.sum(multiplier * log_probs, dim=-1)
        return torch.mean(loss)


class LinearL2DModel_2(nn.Module):
    """
    A pure linear model (equivalent in capacity to a Linear SVM)
    that outputs K+1 scores[cite: 214, 357].
    """

    def __init__(self, input_dim, num_classes=4):
        super(LinearL2DModel_2, self).__init__()
        # No hidden layer, no ReLU. This is a pure linear decision boundary
        self.fc = nn.Linear(input_dim, num_classes + 1)

    def forward(self, x):
        return self.fc(x)


class TrueL2DClassifier_2:
    """
    Wraps the Linear CSS-optimized L2D model with the exact same
    TF-IDF parameters as the baseline LinearSVC.
    """

    def __init__(self, max_features=25000, epochs=15, lr=0.01, batch_size=128, query_cost=0.15):
        # IDENTICAL configuration to your baseline SVM vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=2,
        )
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.query_cost = query_cost
        self.model = None

    def fit(self, train_examples):
        # 1. Transform text data using identical features
        X_train_text = [ex["text"] for ex in train_examples]
        X_train_tfidf = self.vectorizer.fit_transform(X_train_text).toarray()

        y_train = np.array([ex["label"] for ex in train_examples])
        m_train = np.array([simulated_expert_predict(ex) for ex in train_examples])

        # 2. Convert to PyTorch tensors
        X_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        m_tensor = torch.tensor(m_train, dtype=torch.float32)

        # 3. Initialize PyTorch model, optimizer, and loss
        input_dim = X_train_tfidf.shape[1]
        self.model = LinearL2DModel_2(input_dim=input_dim, num_classes=4)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = CSSLoss_2(query_cost=self.query_cost)

        # 4. Training loop (SGD / Adam)
        dataset_size = len(train_examples)
        self.model.train()
        for epoch in range(self.epochs):
            permutation = torch.randperm(dataset_size)
            for i in range(0, dataset_size, self.batch_size):
                indices = permutation[i : i + self.batch_size]
                batch_x = X_tensor[indices]
                batch_y = y_tensor[indices]
                batch_m = m_tensor[indices]

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y, batch_m)
                loss.backward()
                optimizer.step()

    def predict_and_evaluate(self, test_examples):
        if self.model is None:
            raise ValueError(
                "The model must be trained using .fit() before evaluating."
            )

        X_test_text = [ex["text"] for ex in test_examples]
        X_test_tfidf = self.vectorizer.transform(X_test_text).toarray()
        y_true = np.array([ex["label"] for ex in test_examples])
        m_expert = np.array([simulated_expert_predict(ex) for ex in test_examples])

        X_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            # h(x) = argmax g_i(x) [cite: 217, 355]
            raw_predictions = torch.argmax(logits, dim=-1).cpu().numpy()

        # Translate 0-indexed PyTorch predictions back to 1-indexed categories
        model_predictions = raw_predictions + 1

        # Check where prediction corresponds to class K+1 (which represents deferral index 5) [cite: 218]
        deferred_mask = model_predictions == 5

        # Apply routing policy
        team_predictions = np.where(deferred_mask, m_expert, model_predictions)

        deferred_total = int(deferred_mask.sum())
        non_deferred_total = len(test_examples) - deferred_total

        useful_defer = int(
            np.sum(deferred_mask & (model_predictions != y_true) & (m_expert == y_true))
        )
        harmful_defer = int(
            np.sum(deferred_mask & (model_predictions == y_true) & (m_expert != y_true))
        )

        return {
            "policy_name": "True L2D (CSS Loss)",
            "accuracy": accuracy_score(y_true, team_predictions),
            "deferred_total": deferred_total,
            "non_deferred_total": non_deferred_total,
            "useful_defer": useful_defer,
            "harmful_defer": harmful_defer,
            "confusion": build_confusion_rows(y_true, team_predictions),
        }
