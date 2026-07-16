# project3/core/true_l2d.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from .utils import CLASS_IDS, CLASS_NAMES
from .deferral import build_confusion_rows, simulated_expert_predict

class CSSLoss(nn.Module):
    """
    Implements Cost-Sensitive Softmax cross-entropy (CSS) loss.
    """
    def __init__(self):
        super(CSSLoss, self).__init__()

    def forward(self, logits, y_true, m_expert):
        # logits shape: (batch_size, K+1)
        batch_size, num_classes = logits.shape
        K = num_classes - 1  # K standard classes 

        # Compute log-softmax probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # 1. Compute costs c(k) [cite: 372]
        # For standard classes: 1 if y_true != k, else 0 [cite: 372]
        # We represent this cleanly using one-hot structures
        y_one_hot = torch.zeros(batch_size, K, device=logits.device)
        # Shift true labels from 1-based indexing to 0-based indexing
        y_zero_indexed = (y_true - 1).long()
        y_one_hot.scatter_(1, y_zero_indexed.unsqueeze(1), 1.0)
        
        # Cost of predicting normal classes
        c_normal = 1.0 - y_one_hot

        # Cost of deferring: 1 if expert is wrong, else 0 
        c_defer = (y_true != m_expert).float().unsqueeze(1)

        # Combine costs for all K+1 options
        c = torch.cat([c_normal, c_defer], dim=1)

        # 2. Compute multiplier: max_c - c [cite: 371]
        # Since costs are 0 or 1, max_c is 1.0
        multiplier = 1.0 - c

        # 3. Calculate final loss [cite: 371]
        loss = -torch.sum(multiplier * log_probs, dim=-1)
        return torch.mean(loss)


class L2DNeuralNetwork(nn.Module):
    """
    A simple neural network mapping TF-IDF features to K+1 class logits[cite: 214, 218].
    """
    def __init__(self, input_dim, num_classes=4):
        super(L2DNeuralNetwork, self).__init__()
        # Outputs K+1 logits [cite: 214, 218]
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes + 1)  # +1 represents the deferral option 
        )

    def forward(self, x):
        return self.fc(x)


class TrueL2DClassifier:
    """
    Wraps PyTorch training and prediction into a clean scikit-learn styled interface.
    """
    def __init__(self, max_features=10000, epochs=15, lr=0.005, batch_size=64):
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=max_features, min_df=2
        )
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = None

    def fit(self, train_examples):
        # 1. Transform text data
        X_train_text = [ex["text"] for ex in train_examples]
        X_train_tfidf = self.vectorizer.fit_transform(X_train_text).toarray()
        
        y_train = np.array([ex["label"] for ex in train_examples])
        # Generate expert predictions on training data to compute CSS loss [cite: 341]
        m_train = np.array([simulated_expert_predict(ex) for ex in train_examples])

        # 2. Convert to PyTorch tensors
        X_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        m_tensor = torch.tensor(m_train, dtype=torch.float32)

        # 3. Initialize PyTorch model, optimizer and loss
        input_dim = X_train_tfidf.shape[1]
        self.model = L2DNeuralNetwork(input_dim=input_dim, num_classes=4)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = CSSLoss()

        # 4. Training loop [cite: 609]
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
        """
        Executes prediction using the learned decision boundaries.
        The model predicts either class 1..K or defers to the expert (K+1).
        """
        if self.model is None:
            raise ValueError("The model must be trained using .fit() before evaluating.")

        X_test_text = [ex["text"] for ex in test_examples]
        X_test_tfidf = self.vectorizer.transform(X_test_text).toarray()
        y_true = np.array([ex["label"] for ex in test_examples])
        m_expert = np.array([simulated_expert_predict(ex) for ex in test_examples])

        X_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            # Find the argmax across K+1 options [cite: 217, 218]
            raw_predictions = torch.argmax(logits, dim=-1).cpu().numpy()

        # Translate 0-indexed PyTorch class space to Django class keys (1..K)
        model_predictions = raw_predictions + 1 
        
        # Check where prediction corresponds to class K+1 (which represents deferral index 4)
        deferred_mask = (model_predictions == 5)

        # Create ultimate team predictions: if deferred, use expert; else, keep model prediction [cite: 60]
        team_predictions = np.where(deferred_mask, m_expert, model_predictions)
        
        deferred_total = int(deferred_mask.sum())
        non_deferred_total = len(test_examples) - deferred_total

        # Evaluate metrics
        useful_defer = int(np.sum(deferred_mask & (model_predictions != y_true) & (m_expert == y_true)))
        harmful_defer = int(np.sum(deferred_mask & (model_predictions == y_true) & (m_expert != y_true)))

        return {
            "policy_name": "True L2D (CSS Loss)",
            "accuracy": accuracy_score(y_true, team_predictions),
            "deferred_total": deferred_total,
            "non_deferred_total": non_deferred_total,
            "useful_defer": useful_defer,
            "harmful_defer": harmful_defer,
            "confusion": build_confusion_rows(y_true, team_predictions),
        }
    