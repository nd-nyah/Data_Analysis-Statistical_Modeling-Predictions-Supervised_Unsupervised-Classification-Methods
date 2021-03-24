from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # load the datasets as dataframe
    df = {}
    with open('dataset_03_with_header.csv', 'r') as ds:
        df = pd.read_csv(ds)
    print(df.shape)
    print(df.head(2))
    
    # remove rows with missing value and print out the new data shape
    print(df.dropna().shape)
        
    # separate input and target variables
    X = df.drop('y', axis=1)
    y = df.y
    print(X.shape)
    print(y.shape)

    # view target data distribution
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.set_title('target distribution')
    ax1.hist(y)

    # use imputer mean-strategy to fill missing value
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_X = imp.fit_transform(X)
    #print(imp_X)


    # print the statistic summary
    print(pd.DataFrame(imp_X).describe())  

    rs1 = RobustScaler(quantile_range=(25, 75))
    scaled_X = rs1.fit_transform(imp_X)

    # Use a random state for train and test sets split 
    rs = check_random_state(1000)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.25, random_state=rs)

    # use decision tree regressor to train the predictor
    br = DecisionTreeRegressor(criterion='mse', max_depth=11, random_state=1000)
    br.fit(X_train, y_train)

    # training accuracy
    print('Accuracy Score of prediction is: %.3f' % br.score(X_test, y_test))
    print("Accuracy Percentage of prediction is: {:0.2f}%".format(br.score(X_test, y_test) * 100))

    # save model
    joblib.dump(br, 'br_joblib')
    
    # prediction on a test case
    y_predict = br.predict(X_test)

    # write prediction output to a text file
    predict_output = np.array(pd.DataFrame(y_predict))
    pred_file = open('predict_output', 'w')
    for row in predict_output:
        np.savetxt(pred_file, row)
    pred_file.close()

    # evaluation: absolute error and mape with without CV
    print('Non-CV mean absolute error: %.3f' % mean_absolute_error(y_test, y_predict))
    print('Non-CV mean absolute per error: %.3f' % mean_absolute_percentage_error(y_test, y_predict, multioutput = 'raw_values'))

    # evaluation: mean absolute error CV score
    mae_scores = cross_val_score(br, X_test, y_test, cv=10, scoring='neg_mean_absolute_error')
    print('CV mae score mean: %.3f' % mae_scores.mean())
    print('CV mae score std: %.3f' % mae_scores.std())

    # evaluation: mean squared error CV score 
    scores = cross_val_score(br, X_test, y_test, cv=10, scoring='neg_mean_squared_error')
    print('CV Negative mean squared errors mean: %.3f' % scores.mean())
    print('CV Negative mean squared errors std: %.3f' % scores.std())

    # evaluation: CV R2 score
    r2_scores = cross_val_score(br, X_test, y_test, cv=10, scoring='r2')
    print('CV R2 score mean: %.3f' % r2_scores.mean())
    print('CV R2 score std: %.3f' % r2_scores.std())

    # Show the absolute error histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(np.abs(y_predict - y_test), bins='auto', log=True)
    ax.set_xlabel('Absolute error')
    ax.set_ylabel('Sample count')
    ax.grid()

    plt.show()