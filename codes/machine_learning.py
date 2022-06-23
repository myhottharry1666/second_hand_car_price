"""
Name: Hangliang Ren

This program performs many Machine Learning and Deep Learning on given data,
from dataset 'used_car_train_20200313.csv', containing information about used
cars, building model predicting the price of used case based on chosen
attributes, plotting graphs visualizing training process (learning curve),
returning errors of different trained models for future research.
"""
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


class MachineLearning:
    """
    This class includes many functions performing Machine Learning and
    Deep Learning towards the given dataset.
    """
    def __init__(self, featrue_train, feature_test, label_train, label_test):
        """
        post: This function prepares data for Machine Learning (ML).
        """
        self._x_train = featrue_train
        self._x_test = feature_test
        self._y_train = label_train
        self._y_test = label_test

    def linear_regression(self):
        """
        post: Perform linear regression (ML) between chosen attributes and
              price data, training model predicting the price;
              Plot the learning curve;
              Save the plotted curve as 'linear_regression_training_curve.png';
              Save the trained model as 'linear.model';
              Return the mean squared error of trained model.
        """
        # plot the learning curve
        regressor = DecisionTreeRegressor()
        self.plot_learning_curve(regressor, 'Linear Regression',
                                 self._x_train, self._y_train)

        # calculate mean squared error
        regressor = DecisionTreeRegressor()
        regressor.fit(self._x_train, self._y_train)
        predictions = regressor.predict(self._x_test)
        error = mean_squared_error(self._y_test, predictions)
        joblib.dump(regressor, 'linear.model')
        return error / len(self._y_test)

    def ridge_regression(self):
        """
        post: Perform ridge regression (ML) between chosen attributes and
              price data, training model predicting the price;
              Plot the learning curve;
              Save the plotted curve as 'ridge_regression_training_curve.png';
              Save the trained model as 'ridge.model';
              Return the mean squared error of trained model.
        """
        # find a good alpha value (punishment coefficient)
        ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20])
        ridgecv.fit(self._x_train, self._y_train)
        good_a = ridgecv.alpha_

        # plot the learning curve
        model = Ridge(alpha=good_a)
        self.plot_learning_curve(model, 'Ridge Regression',
                                 self._x_train, self._y_train)

        # calculate mean squared error
        model = Ridge(alpha=good_a)
        model.fit(self._x_train, self._y_train)
        predictions = model.predict(self._x_test)
        error = mean_squared_error(self._y_test, predictions)
        joblib.dump(model, 'ridge.model')
        return error / len(self._y_test)

    def random_forest(self):
        """
        post: Perform random forest (ML) between chosen attributes and
              price data, training model predicting the price;
              Plot the learning curve;
              Save the plotted curve as 'random_forest_training_curve.png';
              Save the trained model as 'random_forest.model';
              Return the mean squared error of trained model.
        """
        # plot the learning curve
        model = RandomForestRegressor()
        self.plot_learning_curve(model, 'Random Forest',
                                 self._x_train, self._y_train)

        # calculate mean squared error
        model = RandomForestRegressor()
        model.fit(self._x_train, self._y_train)
        predictions = model.predict(self._x_test)
        error = mean_squared_error(self._y_test, predictions)
        joblib.dump(model, 'random_forest.model')
        return error / len(self._y_test)

    def plot_learning_curve(self, estimator, title, features, labels):
        """
        parameter: estimator, the trained model for plotting training curve;
                   title, the title of the plotted training curve;
                   features, feature data for model training;
                   labels, label data for model training.
        post: Plot learning curve for the input model / estimator;
              Save the plotted curve as 'title_training_curve.png'.
        """
        plt.figure()
        plt.title(title + ' Training Curve')
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, features, labels)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
        plt.legend(loc="best")

        save_name = title.lower().split()
        plt.savefig(save_name[0] + '_' + save_name[1] + '_training_curve.png')
        plt.close('all')

    def neural_l2_sigmoid(self, x_train, x_test, y_train, y_test):
        """
        post: Perform Deep Learning with two-layers neural network,
              l2 normalization, and sigmoid activation,
              training model predicting the price based on chosen attributes;
              Plot the learning curve;
              Save the plotted curve as 'l2_sigmoid_curve.png';
              Return the mean squared error of trained model.
        """
        model = keras.models.Sequential([
            keras.layers.Dense(256, activation='sigmoid'),
            keras.layers.Dense(128, activation='relu',
                               kernel_regularizer=keras.regularizers.l2()),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1)
        ])

        model.compile(loss="mse",
                      optimizer=keras.optimizers.RMSprop(lr=0.1),
                      metrics=["accuracy"])

        history = model.fit(x_train, y_train,
                            batch_size=32,
                            epochs=150,
                            validation_data=(x_test, y_test),
                            validation_freq=1)

        self.plot_neural_learning_curve(history, 'l2_sigmoid_curve.png')
        return history.history['val_loss'][-1]

    def neural_l1_relu(self, x_train, x_test, y_train, y_test):
        """
        post: Perform Deep Learning with two-layers neural network,
              l1 normalization, and relu activation,
              training model predicting the price based on chosen attributes;
              Plot the learning curve;
              Save the plotted curve as 'l1_relu_curve.png';
              Return the mean squared error of trained model.
        """
        model = keras.models.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu',
                               kernel_regularizer=keras.regularizers.l1()),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1)
        ])

        model.compile(loss="mse",
                      optimizer=keras.optimizers.RMSprop(lr=0.1),
                      metrics=["accuracy"])

        history = model.fit(x_train, y_train,
                            batch_size=32,
                            epochs=150,
                            validation_data=(x_test, y_test),
                            validation_freq=1)

        self.plot_neural_learning_curve(history, 'l1_relu_curve.png')
        return history.history['val_loss'][-1]

    def neural_l2_relu(self, x_train, x_test, y_train, y_test):
        """
        post: Perform Deep Learning with two-layers neural network,
              l2 normalization, and relu activation,
              training model predicting the price based on chosen attributes;
              Plot the learning curve;
              Save the plotted curve as 'l2_relu_curve.png';
              Return the mean squared error of trained model.
        """
        model = keras.models.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu',
                               kernel_regularizer=keras.regularizers.l2()),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1)
        ])

        model.compile(loss="mse",
                      optimizer=keras.optimizers.RMSprop(lr=0.1),
                      metrics=["accuracy"])

        history = model.fit(x_train, y_train,
                            batch_size=32,
                            epochs=150,
                            validation_data=(x_test, y_test),
                            validation_freq=1)

        self.plot_neural_learning_curve(history, 'l2_relu_curve.png')
        return history.history['val_loss'][-1]

    def plot_neural_learning_curve(self, history, file_name):
        """
        parameter: history, history training data, including error;
                   file_name, the name of plotted learning curve.
        post: Plot learning curve with history training data;
              Save the plotted curve with input file_name.
        """
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Test Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.savefig(file_name)
        plt.close('all')
