# Second-hand Car Price Prediction

In this project, we use several ML methods (linear regression, ridge regression, random forest, and neural networks) to build models, predicting second-hand car price, based on "used_car_train_20200313.csv" dataset.

This project has the following parts.
1. Data cleaning & Feature engineering
2. Build Machine Learning models (linear regression, ridge regression, random forest)
3. Build Deep Learning models (neural networks)
4. Final analysis and report (**Note: the complete project description, process, results analysis, and charts are in report pdf**)

## Detailed project description

### Files
**codes**
- "data_process.py" do all the Data cleaning & Feature engineering work.
- "machine_learning.py" do all the model training work.
- "main.py" operate all the functions in "data_process.py" and "machine_learning.py".
- "more_test.py" do further comparison among trained models.

**output (plot & results)**
Store all generated plots, trained models parameters, and validation results.

**part2 (final) Report**
Store complete project desciption, process, results analysis, and charts.


### Dataset
Dataset we use: [ used_car_train_20200313.csv](https://drive.google.com/file/d/1fgUXVEAmnvWJVNXFKqOrgHTLkgG66Hoj/view " used_car_train_20200313.csv")

Details about this dataset:
1. 150000 rows/samples, 31 columns/attributes.
2. Attributes in our dataset:
['SaleID', 'name', 'regDate', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'power', 'kilometer', 'notRepairedDamage', 'regionCode', 'seller', 'offerType', 'creatDate', 'price', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']  
Among them, v_0 to v_14 are anonymous variables .
3. The whole dataset has information about 150000 used cars, I will use 100000
for training set, 50000 for test set.

### Data cleaning & Feature engineering
1. Use **normal polynomial interpolation** to fill missing values.
2. Use **heatmap** to identify top 10 attributes most correlated/relevant to the current price
3. Use **pearson correlation coefficient** to compare each individual attribute/column with the price attribute/column, check whether strong correlation exists.
4. Choose 3 attributes/factors from those 10 factors, using **trendline** to draw graphs of each attribute vs. price, observing the correlation.

### Machine Learning
1. Train a **linear regression** model.
2. Train a **ridge regression** model.
3. Train a **random forest** model.
4. Store **mean squared error & learning curve**, compare mean squared error of each model, choose the best model.

### Deep Learning
1. Train a neural net with **two hidden layers**, **sigmoid** as activation function, **L2 normalization** in the loss function, and **learning rate** between **0.1-0.2**.
2. Train a neural net with **two hidden layers**, **relu** as activation function, **L1 normalization** in the loss function, and **learning rate** between **0.1-0.2**.
3. Train a neural net with **two hidden layers**, **relu** as activation function, **L2 normalization** in the loss function, and **learning rate** between **0.1-0.2**.
4. Store **mean squared error & learning curve**, compare mean squared error of each model, choose the best model.

### Final comparison and report
To all the analysis and training described above, their results, charts, and analysis are in report pdf.

## Instructions to use my codes

Please follow steps listed below one by one, from top to bottom.
**Please put all python files, including the dataset, into the same folder!!! (for convenience of description, we call this folderX)**

### Download dataset

Dowload dataset from this link below, and put it in folderX.
**[https://drive.google.com/file/d/1fgUXVEAmnvWJVNXFKqOrgHTLkgG66Hoj/view?usp=sharing](https://drive.google.com/file/d/1fgUXVEAmnvWJVNXFKqOrgHTLkgG66Hoj/view?usp=sharing)**

### Put python files

Put all my submitted python files ('data_process.py', 'machine_learning.py', 'main.py', 'more_test.py') in folderX.

### Required libraries

Open your terminal, use pip to install these libraries:

 1. pandas
 2. scikit-learn
 3. numpy
 4. matplotlib
 5. seaborn
 6. tensorflow

### Path setting

VScode terminal path setting: When operate the code, please use the "Python" terminal, and set the path to the direct folder where you put / store these code files (folderX). See the Red Highlighted part in example given in page 18, part 10a) of my report.


### Run python files

 1. Open and Run main.py
 2. Open and Run more_test.py
 3. Note: make sure all files are in the same folder and the path is set correctly.

##  Meaning of output files

For meaning that each output file represents, please see page 18-19, part 10b) in my report.
