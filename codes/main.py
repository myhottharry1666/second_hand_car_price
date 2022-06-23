"""
Name: Hangliang Ren

This program performs data processing, and data analysis through
Machine Learning & Deep Learning through methods from
'data_process.py' and 'machine_learning.py', on dataset
'used_car_train_20200313.csv', containing information about used cars.
"""
from data_process import DataProcess as DP
from machine_learning import MachineLearning as ML


def main():
    """
    post: Perform data processing, and data analysis through Machine Learning &
          Deep Learning through methods from 'data_process.py' and
          'machine_learning.py', on dataset 'used_car_train_20200313.csv',
          containing information about used cars;
          Save plots for data processing & analysis with names specified in
          'data_process.py' and 'machine_learning.py';
          Save analysis & training data (error) in the file 'result.txt'.
    """
    path = 'used_car_train_20200313.csv'
    data_process = DP(path)
    # do heatmap analysis towards data
    data_process.heatmap_analysis()
    # do pearson coefficient check towards data
    pearson_check = 'pearson check:\n' + str(data_process.pearson_check())
    # do trend line check towards data
    data_process.regression_line_check()
    # get training data for traditional machine learning
    x_train, x_test, y_train, y_test = data_process.train_test_division()
    # get training data for traditional machine learning
    x_train_dp, x_test_dp, y_train_dp, y_test_dp = \
        data_process.process_for_neural()

    machine_learning = ML(x_train, x_test, y_train, y_test)
    # do linear regression
    linear_result = ('linear regression error:\n' +
                     str(machine_learning.linear_regression()))
    # do ridge regression
    ridge_result = ('ridge regression error:\n' +
                    str(machine_learning.ridge_regression()))
    # do random forest
    forest_result = ('random forest error:\n' +
                     str(machine_learning.random_forest()))
    # do deep learning with l2 normalization, sigmoid activation
    error = \
        machine_learning.neural_l2_sigmoid(x_train_dp,
                                           x_test_dp, y_train_dp, y_test_dp)
    e2s_result = ('deep learning with l2 normalization, sigmoid activation ' +
                  'error:\n' + str(error))
    # do deep learning with l1 normalization, relu activation
    error = \
        machine_learning.neural_l1_relu(x_train_dp,
                                        x_test_dp, y_train_dp, y_test_dp)
    e1l_result = ('deep learning with l1 normalization, relu activation ' +
                  'error:\n' + str(error))
    # do deep learning with l2 normalization, relu activation
    error = \
        machine_learning.neural_l2_relu(x_train_dp,
                                        x_test_dp, y_train_dp, y_test_dp)
    e2l_result = ('deep learning with l2 normalization, relu activation ' +
                  'error:\n' + str(error))

    to_writes = [pearson_check, linear_result, ridge_result, forest_result,
                 e2s_result, e1l_result, e2l_result]
    with open('result.txt', 'w') as f:
        for to_write in to_writes:
            f.write("{:-^150s}".format("Split Line"))
            f.write('\n')
            f.write(to_write)
            f.write('\n')
            f.write("{:-^150s}".format("Split Line"))
            f.write('\n\n')


if __name__ == '__main__':
    main()
