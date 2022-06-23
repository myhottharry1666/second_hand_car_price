"""
Name: Hangliang Ren

This program performs more testing for the project.
We in total do two parts of testing:
    1) test the correlation between our chosen attributes and price.
    2) test whether model trained by random forest is better than other
       traditional ML methods.
We have done the part 1) check in 'main.py'; here, we will do the
part 2) check.
"""
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import pandas as pd


def main():
    """
    post: We test our three models trained by three traditional ML methods by
          10000 samples randomly chosen from 'used_car_train_20200313.csv'
          dataset;
          We will calculate both error and accuracy this time;
          Store calculated error and accuracy in 'validation_result.txt'.
    """
    # import models
    linear = joblib.load('linear.model')
    ridge = joblib.load('ridge.model')
    random_forest = joblib.load('random_forest.model')

    # import & prepare data
    path = 'used_car_train_20200313.csv'
    data = pd.read_csv(path, delim_whitespace=True, na_values='-')
    data = data.interpolate(method='polynomial', order=5)
    data = data.sample(10000)
    chosen_attribute = ['regDate', 'model', 'bodyType', 'creatDate', 'v_0',
                        'v_2', 'v_3', 'v_5', 'v_11', 'v_12']
    features = data[chosen_attribute]
    labels = data['price']

    # calculate error & accuracy
    results = []

    linear_acc = linear.score(features, labels)
    prediction = linear.predict(features)
    linear_error = mean_squared_error(labels, prediction) / len(data)
    results.append(('linear', linear_acc, linear_error))

    ridge_acc = ridge.score(features, labels)
    prediction = ridge.predict(features)
    ridge_error = mean_squared_error(labels, prediction) / len(data)
    results.append(('ridge', ridge_acc, ridge_error))

    forest_acc = random_forest.score(features, labels)
    prediction = random_forest.predict(features)
    forest_error = mean_squared_error(labels, prediction) / len(data)
    results.append(('random forest', forest_acc, forest_error))

    # store results in 'validation_result.txt'
    with open('validation_result.txt', 'w') as f:
        for result in results:
            f.write("{:-^150s}".format("Split Line"))
            f.write('\n')
            f.write(result[0] + ':\n')
            f.write('acc: ' + str(result[1]) + '\n')
            f.write('error: ' + str(result[2]) + '\n')
            f.write("{:-^150s}".format("Split Line"))
            f.write('\n\n')


if __name__ == '__main__':
    main()
