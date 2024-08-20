import sklearn.model_selection
from scipy.io import loadmat
import numpy as np
import gradient_descent_logistic
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import save_csv
from sklearn.linear_model import LogisticRegression




# dict_keys(['y', 'X', 'description', 'X_test'])

# description: our features, not normalized.
# ['fixed acidity       ' 'volatile acidity    ' 'citric acid         '
#  'residual sugar      ' 'chlorides           ' 'free sulfur dioxide '
#  'total sulfur dioxide' 'density             ' 'pH                  '
#  'sulphates           ' 'alcohol             ' 'quality             ']

# X: Training data: (5000, 12)
# y: training labels: 0 or 1, (5000, 1)
# X_test: test data, (1000, 12)

def normalize_data(data, mean = None, std = None):
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
    std = np.where(std == 0, 1 , std)
    data = (data - mean) / std
    return data, mean, std

def add_fictitious_dimension(data):
    ones_array = np.ones((data.shape[0], 1))
    return np.hstack((data, ones_array))



# Grid Search For learning_rate and lambda
def GridSearch(param_grid):

    """
    :param  param_grid = {
        'learning_rate': [0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5],
        'lambda': [1e-7, 1e-6, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10]
    }
    """

    model = gradient_descent_logistic.LogisticRegressionModel(max_iterations=1000, precision=1e-4)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(wine_training, training_labels)

    print(grid_search.best_params_)

    return grid_search.best_estimator_


if __name__ == '__main__':
    wine_data = loadmat('./data.mat')
    normalized_wine_training, mean_training, std_training = normalize_data(wine_data['X'])
    normalized_test, _, _ = normalize_data(wine_data['X_test'], mean_training, std_training)
    normalized_test = add_fictitious_dimension(normalized_test)
    normalized_wine_training = add_fictitious_dimension(normalized_wine_training)
    nb_samples, nb_features = normalized_wine_training.shape
    wine_training, wine_validation, training_labels, validation_labels = sklearn.model_selection.train_test_split(
        normalized_wine_training, wine_data['y'].reshape(wine_data['y'].shape[0]), train_size=0.8)


    # Batch Gradient Descent Cost visualization
    lambda_param = 1e-7
    step_size = 0.001
    num_iter = 7000

    model_GD =  gradient_descent_logistic.LogisticRegressionModel(step_size, num_iter, lambda_param)
    clf = model_GD.fit(normalized_wine_training, wine_data['y'].ravel(), method='BatchGradient')
    save_csv.results_to_csv(clf.predict(normalized_test))
    plt.plot(np.arange(len(clf.cost_history)), clf.cost_history)
    plt.xlabel('Number of iterations in training')
    plt.ylabel('Cost at the end of training')
    plt.title('Training loss vs. Number of iterations for Batch Gradient Descent')
    plt.savefig('Wine_GD.png')
    plt.show()


    # Stochastic Gradient Descent Without Decay Cost visualization
    stoch_lambda = 0.001
    stoch_step_size = 0.04
    model_SGD = gradient_descent_logistic.LogisticRegressionModel(stoch_step_size, num_iter, stoch_lambda)
    clf_sgd = model_SGD.fit(wine_training, training_labels, method='StochGradient')
    plt.plot(np.arange(len(clf_sgd.cost_history)), clf_sgd.cost_history)
    plt.xlabel('Number of iterations in training')
    plt.ylabel('Cost at the end of training')
    plt.title('Training loss vs. Number of iterations for Stochastic Gradient Descent')
    plt.savefig('Wine_SGD.png')
    plt.show()

    # Stochastic Gradient Descent With Decay Cost visualization

    model_decay = gradient_descent_logistic.LogisticRegressionModel(stoch_step_size, num_iter, stoch_lambda)
    clf_decay = model_decay.fit(wine_training, training_labels, method='StochGradientDecay')
    plt.plot(np.arange(len(clf_decay.cost_history)), clf_decay.cost_history, label='Decay')
    plt.plot(np.arange(len(clf_sgd.cost_history)), clf_sgd.cost_history, label='No Decay')
    plt.xlabel('Number of iterations in training')
    plt.ylabel('Cost at the end of training')
    plt.legend()
    plt.title('Training loss vs. Number of iterations for SGD W/Without Decay')
    plt.savefig('Wine_SGD_NoDecay_VS_Decay.png')
    plt.show()



    # Training Vs. Validation for SGD and GD

    # As Expected GD preforms better.

    training_splits = [100, 500, 1000, 3000, 4000]
    training_scores_GD = []
    validation_scores_GD = []
    training_scores_SGD = []
    validation_scores_SGD = []
    model_GD =  gradient_descent_logistic.LogisticRegressionModel(step_size, num_iter, lambda_param)
    model_SGD = gradient_descent_logistic.LogisticRegressionModel(stoch_step_size, num_iter, stoch_lambda)
    for split in training_splits:
        clf_GD = model_GD.fit(wine_training[:split] ,training_labels[:split] , method='BatchGradient')
        clf_SGD = model_SGD.fit(wine_training[:split] ,training_labels[:split] , method='StochGradient')

        training_scores_GD.append(clf_GD.score(wine_training[:split] ,training_labels[:split]))
        validation_scores_GD.append(clf_GD.score(wine_validation, validation_labels))

        training_scores_SGD.append(clf_SGD.score(wine_training[:split], training_labels[:split]))
        validation_scores_SGD.append(clf_SGD.score(wine_validation, validation_labels))

    plt.plot(training_splits, training_scores_GD, label='Training-GD')
    plt.plot(training_splits, validation_scores_GD, label='Validation_GD')

    plt.plot(training_splits, training_scores_SGD, label='Training-SGD')
    plt.plot(training_splits, validation_scores_SGD, label='Validation_SGD')

    plt.xlabel('Number of samples in training')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Training and validation accuracy vs. Number of samples for SGD/GD')
    plt.savefig('Wine_Validation_Training.png')
    plt.show()
