import pandas as pd
import numpy as np
import math

#Imports for creating visualizations
import matplotlib.pyplot as plt 
import seaborn as sns

#Imports for modeling and evaluation
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE

# Feature Engineering/Selection

def select_rfe(X_train_scaled, y_train, k, return_rankings=False, model=LinearRegression()):
    # Use the passed model, LinearRegression by default
    rfe = RFE(model, n_features_to_select=k)
     # fit the data using RFE
    rfe.fit(X_train_scaled, y_train)
    # get mask of columns selected as list
    feature_mask = X_train_scaled.columns[rfe.support_].tolist()
    if return_rankings:
        rankings = pd.Series(dict(zip(X_train_scaled.columns, rfe.ranking_)))
        return feature_mask, rankings
    else:
        return feature_mask

## evaluation functions

def residuals(actual, predicted):
    '''
    ∆(y,yhat)
    '''
    return actual - predicted

def sse(actual, predicted):
    '''
    sum of squared error
    '''
    return (residuals(actual, predicted) ** 2).sum()

def mse(actual, predicted):
    '''
    mean squared error
    '''
    n = actual.shape[0]
    return sse(actual, predicted) / n

def rmse(actual, predicted):
    '''
    root mean squared error
    '''
    return math.sqrt(mse(actual, predicted))

def ess(actual, predicted):
    '''
    explained sum of squared error
    '''
    return ((predicted - actual.mean()) ** 2).sum()

def tss(actual):
    '''
    total sum of squared error
    '''
    return ((actual - actual.mean()) ** 2).sum()

def r2_score(actual, predicted):
    '''
    explained variance
    '''
    return ess(actual, predicted) / tss(actual)


## Evaluating model against baseline model functions

def regression_errors(actual, predicted):
    return pd.Series({
        'sse': sse(actual, predicted),
        'ess': ess(actual, predicted),
        'tss': tss(actual),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
        'r2': r2_score(actual, predicted),
    })

def baseline_mean_errors(actual):
    predicted = actual.mean()
    return {
        'sse': sse(actual, predicted),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    }

def better_than_baseline(actual, predicted):
    sse_baseline = sse(actual, actual.mean())
    sse_model = sse(actual, predicted)
    return sse_model < sse_baseline

### evaluation that plots residuals 

def plot_residuals(actual, predicted, data):
    residuals = actual - predicted
    plt.figure(figsize = (9,6), facecolor="lightblue")
    sns.set_theme(style="whitegrid")
    sns.scatterplot(x = actual, y = residuals, data = data, color="blue")
    plt.axhline(0, ls = ':')
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual', fontsize = 15)
    plt.show()

    
## Functions for final notebook

def viz1(predictions_test):
    plt.figure(figsize = (16, 8))
    plt.plot(predictions_test.actual, predictions_test.actual, color = 'green')
    plt.plot(predictions_test.actual, predictions_test.le_pred_mean, color = 'red')
    plt.scatter(predictions_test.actual, predictions_test.le_pred_poly_rfe, alpha = 0.5)
    plt.annotate('Ideal: Actual Life Expectancy', (1.2*10**6, 1.25*10**6), rotation = 25)
    plt.annotate('Baseline: Mean Life Expectancy', (1.25*10**6, .3*10**6), color = 'red')
    plt.title('Visualizing Polynomial RFE Model on Test', fontsize=20)
    plt.show()
    
def viz2(predictions_test):
# plot to visualize actual vs predicted. 
    plt.figure(figsize=(16,8))
    plt.hist(predictions_test.actual, color='blue', alpha=.5, label="Actual Assessed Value")
    plt.hist(predictions_test.le_pred_poly_rfe, color='red', alpha=.5, label="Model: Polynomial RFE Model")
    plt.xlabel("Life Expectancy Distribution")
    plt.ylabel("Count")
    plt.title("Comparing Actual Life Expectancy vs. Predicted Values for the Top Model", fontsize=20)
    plt.legend()
    plt.show()