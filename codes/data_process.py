"""
Name: Hangliang Ren

This program performs many analysis on dataset 'used_car_train_20200313.csv',
containing information about used cars.
This program performs analysis mainly through plotting graphs visualizing
correlation between different factors and price, checking correlation
through different coefficients and plots.
This program also divides datasets used for future machine learning.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class DataProcess:
    """
    This class includes many functions performing data processing,
    data analysis, and dataset division towards the given dataset.
    """
    def __init__(self, path):
        """
        post: This function does some preprocessing to the given dataset,
              filling missing values.
        """
        data = pd.read_csv(path, delim_whitespace=True, na_values='-')
        data = data.interpolate(method='polynomial', order=5)
        self._data = data

    def heatmap_analysis(self):
        """
        post: Perform the analysis towards the dataset through heatmap;
              Plot the heapmap of the dataset;
              Save the plotted heatmap as 'heatmap_for_correlation.png'.
        """
        corrmat = self._data.corr()
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, cmap="YlGnBu", square=True)
        plt.title('Heatmap For Correlation')
        plt.savefig('heatmap_for_correlation.png')
        plt.close('all')

    def pearson_check(self):
        """
        post: Check correlation between chosen attributes and price with
              pearson coefficient;
              Return the calculated pearson coefficient.
        """
        chosen_attribute = ['regDate', 'model', 'bodyType', 'creatDate', 'v_0',
                            'v_2', 'v_3', 'v_5', 'v_11', 'v_12']
        corr_results = []

        for attribute in chosen_attribute:
            sub_data = self._data[[attribute, 'price']]
            corr_result = sub_data.corr(method='pearson')
            corr_result = corr_result.loc[attribute, 'price']
            corr_results.append((attribute, corr_result))

        return corr_results

    def regression_line_check(self):
        """
        post: Check correlation between chosen attributes and price with
              trend / regression line;
              Plot trend / regression line between each chosen attribute and
              price;
              Save each plotted trend / regression line as
              '(attribute_name)_and_price_trend.png'.
        """
        chosen_attribute = ['regDate', 'model', 'bodyType', 'creatDate', 'v_0',
                            'v_2', 'v_3', 'v_5', 'v_11', 'v_12']

        for attribute in chosen_attribute:
            sub_data = self._data.sample(300)
            sns.regplot(x=attribute, y="price", data=sub_data)
            plt.title(attribute.capitalize() + ' And Price Trend Line')
            plt.savefig(attribute + '_and_price_trend.png')
            plt.close('all')

    def train_test_division(self):
        """
        post: Prepare data for Machine Learning;
              Divide the dataset into training and test datasets;
              Return training features, test features, training labels,
              test labels.
        """
        chosen_attribute = ['regDate', 'model', 'bodyType', 'creatDate', 'v_0',
                            'v_2', 'v_3', 'v_5', 'v_11', 'v_12']
        sub_data_features = self._data[chosen_attribute]
        sub_data_labels = self._data['price']

        feature_train, feature_test, label_train, label_test = \
            train_test_split(sub_data_features, sub_data_labels, test_size=1/3)
        return feature_train, feature_test, label_train, label_test

    def process_for_neural(self):
        """
        post: Prepare data for Deep Learning;
              Choose & Divide the dataset into training and test datasets;
              Return training features, test features, training labels,
              test labels.
        """
        sub_data = self._data.sample(10000)
        chosen_attribute = ['regDate', 'creatDate', 'v_0', 'v_2', 'v_3',
                            'v_12']
        sub_data_features = sub_data[chosen_attribute]
        sub_data_labels = sub_data['price']
        feature_train, feature_test, label_train, label_test = \
            train_test_split(sub_data_features, sub_data_labels, test_size=1/5)

        mean = feature_train.mean(axis=0)
        std = feature_train.std(axis=0)
        feature_train -= mean
        feature_train /= std
        feature_test -= mean
        feature_test /= std

        return feature_train, feature_test, label_train, label_test
