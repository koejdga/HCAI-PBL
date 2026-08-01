# project3/core/true_l2d.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from .utils import CLASS_IDS
from .deferral import build_confusion_rows, simulated_expert_predict, predict_expert, get_expert_cost


class CSSLoss(nn.Module):
    """
    Implements generalized Cost-Sensitive Softmax cross-entropy (CSS) loss
    supporting 1 or more experts with explicit per-expert query cost parameters.
    """

    def __init__(self, query_cost=0.15):
        super(CSSLoss, self).__init__()
        self.query_cost = query_cost

    def forward(self, logits, y_true, m_experts):
        # logits shape: (batch_size, K + num_experts)
        # m_experts shape: (batch_size, num_experts)
        batch_size, num_outputs = logits.shape
        num_experts = m_experts.shape[1]
        K = num_outputs - num_experts  # Number of baseline classification classes

        log_probs = torch.log_softmax(logits, dim=-1)

        # 1. Get true label one-hot matrix for normal classes
        y_one_hot = torch.zeros(batch_size, K, device=logits.device)
        y_zero_indexed = (y_true - 1).long()
        y_one_hot.scatter_(1, y_zero_indexed.unsqueeze(1), 1.0)

        # 2. Compute individual cost profiles for all options
        # Standard classes cost: 1 if prediction is wrong, else 0
        c_normal = 1.0 - y_one_hot

        # Expert classes cost: 1 if expert is wrong, else 0, plus the query penalty
        if isinstance(self.query_cost, (list, tuple, np.ndarray, torch.Tensor)):
            costs_tensor = torch.tensor(self.query_cost, dtype=torch.float32, device=logits.device).view(1, num_experts)
        else:
            costs_tensor = torch.tensor([self.query_cost] * num_experts, dtype=torch.float32, device=logits.device).view(1, num_experts)

        y_true_expanded = y_true.unsqueeze(1).expand(-1, num_experts)
        c_experts = (y_true_expanded != m_experts).float() + costs_tensor

        # Concatenate costs across all choices
        c = torch.cat([c_normal, c_experts], dim=1)

        # 3. Compute cost-sensitive target multiplier matrix (max_c - c)
        max_c, _ = torch.max(c, dim=1, keepdim=True)
        multiplier = max_c - c

        # 4. Calculate final expected loss
        loss = -torch.sum(multiplier * log_probs, dim=-1)
        return torch.mean(loss)


class L2DNeuralNetwork(nn.Module):
    """
    A simple neural network mapping features to K + num_experts logits.
    """

    DEFAULT_EPOCH_COUNT = 10
    DEFAULT_LEARNING_RATE = 0.005
    DEFAULT_BATCH_SIZE = 64

    def __init__(self, input_dim, num_classes=4, num_experts=1):
        super(L2DNeuralNetwork, self).__init__()
        # Dynamic output sizing mapping to either K+1 or K+2 classes
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes + num_experts),
        )

    def forward(self, x):
        return self.fc(x)


class L2DLinearModel(nn.Module):
    """
    A pure linear model (equivalent to a Linear SVM) outputting K + num_experts scores.
    """

    DEFAULT_EPOCH_COUNT = 6
    DEFAULT_LEARNING_RATE = 0.01
    DEFAULT_BATCH_SIZE = 128

    def __init__(self, input_dim, num_classes=4, num_experts=1):
        super(L2DLinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes + num_experts)

    def forward(self, x):
        return self.fc(x)


class L2DClassifier:
    """Learning to Defer Classifier handling single or multi-expert joint training."""

    def __init__(
        self, model_type, policy_name, tfidf_max_features=10000, query_cost=0.15
    ):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=tfidf_max_features,
            min_df=2,
        )
        self.model_type = model_type
        self.policy_name = policy_name
        self.epochs = model_type.DEFAULT_EPOCH_COUNT
        self.lr = model_type.DEFAULT_LEARNING_RATE
        self.batch_size = model_type.DEFAULT_BATCH_SIZE
        self.query_cost = query_cost
        self.model = None

    def _extract_expert_predictions(self, examples, expert_configs=None):
        """Helper to collect a 2D array of predictions across all available experts."""
        if not expert_configs:
            preds = [simulated_expert_predict(ex) for ex in examples]
            return np.array(preds).reshape(-1, 1)

        # Iterate over provided runtime expert system dictionary profiles
        all_preds = []
        for config in expert_configs:
            preds = [
                predict_expert(ex, config)
                for ex in examples
            ]
            all_preds.append(preds)
        return np.column_stack(all_preds)

    def fit(self, train_examples, expert_configs=None):
        # 1. Transform text data
        X_train_text = [ex["text"] for ex in train_examples]
        X_train_tfidf = self.vectorizer.fit_transform(X_train_text).toarray()

        y_train = np.array([ex["label"] for ex in train_examples])
        m_train = self._extract_expert_predictions(train_examples, expert_configs)
        num_experts = m_train.shape[1]

        # 2. Convert to PyTorch tensors
        X_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        m_tensor = torch.tensor(m_train, dtype=torch.float32)

        # 3. Initialize dynamic multi-output network architecture
        input_dim = X_train_tfidf.shape[1]
        self.model = self.model_type(
            input_dim=input_dim, num_classes=len(CLASS_IDS), num_experts=num_experts
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        if expert_configs:
            query_costs = [get_expert_cost(cfg) for cfg in expert_configs]
        else:
            query_costs = self.query_cost
        criterion = CSSLoss(query_costs)

        # 4. Training loop
        dataset_size = len(train_examples)
        self.model.train()
        for _ in range(self.epochs):
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

    def predict_and_evaluate(self, test_examples, expert_configs=None):
        if self.model is None:
            raise ValueError(
                "The model must be trained using .fit() before evaluating."
            )

        X_test_text = [ex["text"] for ex in test_examples]
        X_test_tfidf = self.vectorizer.transform(X_test_text).toarray()
        y_true = np.array([ex["label"] for ex in test_examples])

        m_experts = self._extract_expert_predictions(test_examples, expert_configs)
        K = len(CLASS_IDS)

        X_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            # Find the argmax index across normal classes and deferral slots
            raw_predictions = torch.argmax(logits, dim=-1).cpu().numpy()

        # Build dynamic team target resolutions
        team_predictions = np.zeros(len(test_examples))
        deferred_mask = np.zeros(len(test_examples), dtype=bool)

        useful_defer = 0
        harmful_defer = 0
        total_cost = 0.0

        if expert_configs:
            expert_costs = [get_expert_cost(cfg) for cfg in expert_configs]
        else:
            expert_costs = [self.query_cost]

        num_experts = len(expert_costs)
        query_allocation = {
            cid: {e_idx: {"correct": 0, "queried": 0} for e_idx in range(num_experts)}
            for cid in CLASS_IDS
        }

        for idx in range(len(test_examples)):
            pred_class = raw_predictions[idx]

            if pred_class < K:
                # Model confidently handled classification internally (1-based label alignment)
                team_predictions[idx] = pred_class + 1
            else:
                # System opted to defer to an expert target channel!
                deferred_mask[idx] = True
                expert_idx = (
                    pred_class - K
                )  # Maps target slot directly back to expert index
                chosen_expert_pred = m_experts[idx, expert_idx]
                team_predictions[idx] = chosen_expert_pred
                if expert_idx < len(expert_costs):
                    total_cost += expert_costs[expert_idx]
                else:
                    total_cost += expert_costs[0]

                true_label = int(y_true[idx])
                allocated_idx = expert_idx if expert_idx < num_experts else 0
                query_allocation[true_label][allocated_idx]["queried"] += 1
                if chosen_expert_pred == true_label:
                    query_allocation[true_label][allocated_idx]["correct"] += 1

                # Fallback calculation logic modeling solo classifier counterfactual baseline
                # Obtain a standalone proxy estimation from standard internal text prediction classes
                fallback_class_pred = torch.argmax(logits[idx][:K]).item() + 1

                if (
                    fallback_class_pred != y_true[idx]
                    and chosen_expert_pred == y_true[idx]
                ):
                    useful_defer += 1
                elif (
                    fallback_class_pred == y_true[idx]
                    and chosen_expert_pred != y_true[idx]
                ):
                    harmful_defer += 1

        deferred_total = int(deferred_mask.sum())
        non_deferred_total = len(test_examples) - deferred_total

        return {
            "policy_name": self.policy_name,
            "accuracy": accuracy_score(y_true, team_predictions),
            "deferred_total": deferred_total,
            "non_deferred_total": non_deferred_total,
            "useful_defer": useful_defer,
            "harmful_defer": harmful_defer,
            "total_cost": round(total_cost, 2),
            "query_allocation": query_allocation,
            "confusion": build_confusion_rows(y_true, team_predictions),
        }
