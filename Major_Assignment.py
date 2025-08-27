# Logistic Regression

import numpy as np
import csv

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X, threshold=0.5):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted_probs = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > threshold else 0 for i in y_predicted_probs]
        return np.array(y_predicted_cls)

def load_data_from_csv(filename):
    X = []
    y = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            label = 1 if row[1] == 'M' else 0
            y.append(label)
            features = [float(value) for value in row[2:]]
            X.append(features)
    return np.array(X), np.array(y)

def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    shuffled_indices = np.random.permutation(n_samples)
    test_set_size = int(n_samples * test_size)
    
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X, y = load_data_from_csv('data.csv')
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=123)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    regressor = LogisticRegression(learning_rate=0.1, n_iters=1000)
    regressor.fit(X_train, y_train)
    
    predictions = regressor.predict(X_test)

    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")

# Decision Tree


import numpy as np
from collections import Counter
import csv

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left_child = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feat, best_thresh, left_child, right_child)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        current_impurity = self._gini(y)

        for feat_idx in feat_idxs:
            thresholds = np.unique(X[:, feat_idx])
            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X[:, feat_idx], threshold)

                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                n = len(y)
                n_l, n_r = len(left_idxs), len(right_idxs)
                g_l, g_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
                weighted_gini = (n_l / n) * g_l + (n_r / n) * g_r
                
                info_gain = current_impurity - weighted_gini

                if info_gain > best_gain:
                    best_gain = info_gain
                    split_idx = feat_idx
                    split_thresh = threshold
                    
        return split_idx, split_thresh

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, y):
        hist = np.bincount(y.astype(int))
        ps = hist / len(y)
        return 1 - np.sum([p**2 for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

def load_data_from_csv(filename):
    X = []
    y = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            label = 1 if row[1] == 'M' else 0
            y.append(label)
            features = [float(value) for value in row[2:]]
            X.append(features)
    return np.array(X), np.array(y)

def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    shuffled_indices = np.random.permutation(n_samples)
    test_set_size = int(n_samples * test_size)
    
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X, y = load_data_from_csv('data.csv')
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=123)

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")
