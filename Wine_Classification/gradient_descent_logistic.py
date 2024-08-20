import numpy as np
from sklearn.linear_model import LogisticRegression
class LogisticRegressionModel:

    # In my implementation I followed the sklearn estimator interface, so I can use
    # GridSearchCV and other helpful functions for hyperparameter tuning.
    def __init__(self, learning_rate=0.01, max_iterations=1000, lambda_param = 1.0, precision = 0.0001):
        self.nb_features = None
        self.nb_samples = None
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.precision = precision
        self.max_iterations = max_iterations
        self.W = None
        self.cost_history = []
    @staticmethod
    def sigmoid(x):
        return np.exp(-np.logaddexp(0, -x))

    def _loss(self, h,y, W):
        np.seterr(divide='ignore')
        return ((-np.dot(y, np.log(h)) - np.dot((1 - y), np.log(1 - h))) + self.lambda_param
                * np.sum(np.square(W)))
    def _batch_gradient(self, X, y , initial_W):
        W = initial_W
        for epoch in range(self.max_iterations):
            predicted_prob = self.sigmoid(np.dot(X, W))

            # I'm penalizing the intercept since after testing it turned out to give better results, our bias term
            # is more likely to be 0, https://stats.stackexchange.com/questions/376259/penalize-the-intercept-in-lasso-l1-penalized-logistic-regression-or-not

            # In the gradient update there some constants I got rid of and assumed would be regularized with
            # lambda, as they do not affect the minimum of our objective

            gradient_update = (X.T @ (predicted_prob - y)
                               + self.lambda_param * W)

            W -= self.learning_rate * gradient_update
            if np.linalg.norm(gradient_update) < self.precision:
                break

            self.cost_history.append(self._loss(predicted_prob, y, W))
        self.W = W

    # Stochastic gradient gives worse results than batch gradient, but it's a fair estimate.
    # With decay, we converge rapidly at first, but then it stagnates at a value higher than the value yielded
    # Without decay.

    def _stochastic_gradient(self, X, y, initial_W, decay, decay_rate):
        i = 0
        W = initial_W
        shuffled_X = X
        shuffled_y = y
        step_size = self.learning_rate

        for epoch in range(self.max_iterations):

            # There are some conflicting definitions of SGD, I stuck with this one, we assume the points
            # follow a uniform distribution, this yielded the best results

            if i == self.nb_samples:
                indices = np.random.permutation(self.nb_samples)
                shuffled_X = shuffled_X[indices]
                shuffled_y = shuffled_y[indices]
                i = 0

            predicted_prob = self.sigmoid(np.dot(shuffled_X, W))
            gradient_update = ((predicted_prob[i] - shuffled_y[i]) * shuffled_X[i]
                               + self.lambda_param * W)
            if decay:
                step_size = step_size / (1 + decay_rate * epoch)

            W -= step_size * gradient_update

            if np.linalg.norm(gradient_update) < self.precision:
                print("Convergence reached")
                break

            self.cost_history.append(self._loss(predicted_prob, shuffled_y, W))
            i += 1  # Increment i here
        self.W = W



    def fit(self, X, y, method = 'BatchGradient', decay_rate = 0.001):
        self.nb_samples, self.nb_features = X.shape
        initial_W = np.zeros(self.nb_features)
        if method == 'BatchGradient':
            self.cost_history = []
            self._batch_gradient(X, y, initial_W)
        elif method == 'StochGradient':
            self.cost_history = []
            self._stochastic_gradient(X, y, initial_W, False, decay_rate)
        elif method == 'StochGradientDecay':
            self.cost_history = []
            self._stochastic_gradient(X, y, initial_W, True, decay_rate)
        else:
            raise RuntimeError('You must pass a valid optimization method')
        return self

    def predict_prob(self, X):
        return self.sigmoid(np.dot(X, self.W))

    def predict(self, X):
        return (self.predict_prob(X) > 0.5).astype(int)

    def get_params(self, deep = True):
        return {
            'learning_rate':self.learning_rate,
            'lambda_param': self.lambda_param,
            'max_iterations': self.max_iterations,
            'precision': self.precision
        }
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def score(self, X, y):
        y_predictions = self.predict(X)
        return np.mean(y_predictions == y)
