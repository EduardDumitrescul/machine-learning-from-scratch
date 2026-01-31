import numpy as np
from util.errors import mean_squared_error, coefficient_of_correlation


class DecisionTreeRegressorNode:
    def __init__(self, min_samples_leaf=5, min_samples_split=20, max_features=None):
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.best_feature_index = None
        self.best_threshold = None
        self.prediction = None
        self.left = None
        self.right = None

    def fit(self, x, y):
        if x.shape[0] < self.min_samples_split:
            self.prediction = np.mean(y)
            return

        num_features = x.shape[1]
        best_mse = np.inf
        indexes = np.arange(0, num_features)
        if self.max_features is not None:
            indexes = np.random.choice(indexes, self.max_features, replace=False)
        for feature_index in indexes:
            unique_values = np.sort(np.unique(x[:, feature_index]))
            for i in range(self.min_samples_leaf, unique_values.shape[0]-self.min_samples_leaf-1):
                threshold = (unique_values[i+1] + unique_values[i]) / 2

                x_left, y_left, x_right, y_right = self._split_by_threshold(x, y, feature_index, threshold)

                mean_left = np.mean(y_left)
                mean_right = np.mean(y_right)

                mse_left = mean_squared_error(y_left, mean_left)
                mse_right = mean_squared_error(y_right, mean_right)

                mse = (mse_left * len(y_left) + mse_right * len(y_right)) / (len(y_left) + len(y_right))
                if best_mse > mse:
                    best_mse = mse
                    self.best_feature_index = feature_index
                    self.best_threshold = threshold

        if self.best_feature_index is None or self.best_threshold is None:
            self.prediction = np.mean(y)
            return

        x_left, y_left, x_right, y_right = self._split_by_threshold(x, y, self.best_feature_index, self.best_threshold)
        print(f"Node split: feature_index={self.best_feature_index}, threshold={self.best_threshold}, left_size={x_left.shape[0]}, right_size={x_right.shape[0]}")
        self.left = DecisionTreeRegressorNode(
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split
        )
        self.left.fit(x_left, y_left)
        self.right = DecisionTreeRegressorNode(
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split
        )
        self.right.fit(x_right, y_right)

    def predict(self, x):
        if self.prediction is not None:
            return self.prediction

        if x[self.best_feature_index] <= self.best_threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    @staticmethod
    def _split_by_threshold(x, y, feature_index, threshold):
        x_left, y_left = x[x[:, feature_index] <= threshold], y[x[:, feature_index] <= threshold]
        x_right, y_right = x[x[:, feature_index] > threshold], y[x[:, feature_index] > threshold]
        return x_left, y_left, x_right, y_right

class CustomDecisionTreeRegressor:
    def __init__(
            self,
            min_samples_leaf=5,
            min_samples_split=20,
            max_features = None
    ):
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.root = DecisionTreeRegressorNode(
            min_samples_leaf = self.min_samples_leaf,
            min_samples_split = self.min_samples_split,
            max_features = max_features
        )

    def fit(self, x, y):
        self.root.fit(x, y)

    def predict(self, x):
        return self.root.predict(x)

    def score(self, x, y):
        y_pred = [self.predict(x) for x in x]
        return coefficient_of_correlation(y, y_pred)