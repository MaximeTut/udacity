'''
A module to hold the churn_library functions

Author: Maxime
Date: May 15, 2022
'''


# import libraries
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

PATH_DF = "data\bank_data.csv"
EDA_PATH = "images\eda"
REPORT_PATH = "images\\autre"

def import_data(path):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataset: pandas dataframe
    '''
    data = pd.read_csv(path, index_col=[0])
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data


def perform_eda(dataset):
    '''
    perform eda on dataset and save figures to images folder
    input:
            dataset: pandas dataframe

    output:
            None
    '''
    size = {"fontsize": 15}

    plt.figure(figsize=(8, 8))
    dataset.Churn.value_counts('normalize').plot(kind='bar')
    plt.title("Percentage of churning", fontdict=size)
    plt.savefig(Path(EDA_PATH, "churn_rate.png"))

    plt.figure(figsize=(8, 8))
    dataset['Customer_Age'].hist()
    plt.title("Age distribution", fontdict=size)
    plt.savefig(Path(EDA_PATH, "customer_age.png"))

    plt.figure(figsize=(8, 8))
    dataset.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title("Percentage of Marital status", fontdict=size)
    plt.savefig(Path(EDA_PATH, "Marital_status.png"))

    plt.figure(figsize=(8, 8))
    sns.distplot(dataset['Total_Trans_Ct'], kde=False)
    plt.title("Distribution of Total_Trans_Ct", fontdict=size)
    plt.savefig(Path(EDA_PATH, "Total_Trans_Ct.png"))

    plt.figure(figsize=(8, 8))
    sns.heatmap(dataset.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title("Correlation matrice", fontdict=size)
    plt.savefig(Path(EDA_PATH, "Correlation.png"))


def encoder_helper(dataset):
    '''
    helper function to turn each categorical column into\
    a new column with propotion of churn for each category

    input:
            dataset: pandas dataframe
            category_lst: list of columns that contain categorical features
            used for naming variables or index y column]

    output:
            dataset: pandas dataframe with new columns for
    
    '''
    category_list = dataset.select_dtypes(["object"]).columns

    for variable in category_list:
        var_mean = dataset.groupby(variable).mean()['Churn']
        var_mean_mapping_mean = var_mean.to_dict()
        dataset[variable] = dataset[variable].map(var_mean_mapping_mean)

    return dataset


def perform_feature_engineering(dataset):
    '''
    input:
    dataset: pandas dataframe
    that could be used for naming variables or index y column
    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    dataset = encoder_helper(dataset)

    feats = dataset.drop(["Attrition_Flag", "CLIENTNUM", "Churn"], axis=1)
    target = dataset["Churn"]
    x_train, x_test, y_train, y_test = train_test_split(
        feats, target, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(
                                y_train_preds_logisticregression,
                                y_train_preds_random_forest,
                                y_test_preds_logisticregression,
                                y_test_preds_random_forest):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_random_forest)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_random_forest)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(Path(REPORT_PATH, "report_rf.png"))

    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_logisticregression)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_logisticregression)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(Path(REPORT_PATH, "report_lr.png"))


def feature_importance_plot(randomforest, features):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            path: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = randomforest.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [features.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(features.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(features.shape[1]), names, rotation=90)
    plt.savefig(Path(REPORT_PATH, "features_importance.png"))


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    random_forest = RandomForestClassifier(random_state=42)
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=3000)

    random_forest.fit(x_train, y_train)
    logistic_regression.fit(x_train, y_train)

    joblib.dump(random_forest, './models/randomForest_model.pkl')
    joblib.dump(logistic_regression, './models/logistic_model.pkl')

    feature_importance_plot(
        random_forest,
        x_test)


    y_train_preds_logisticregression = logistic_regression.predict(x_train)
    y_train_preds_random_forest = random_forest.predict(x_train)
    y_test_preds_logisticregression = logistic_regression.predict(x_test)
    y_test_preds_random_forest = random_forest.predict(x_test)

    classification_report_image(
                                y_train_preds_logisticregression,
                                y_train_preds_random_forest,
                                y_test_preds_logisticregression,
                                y_test_preds_random_forest)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(random_forest, x_test, y_test, ax=ax, alpha=0.8)
    lrc_plot = plot_roc_curve(logistic_regression, x_test, y_test, ax = ax)
    plt.savefig(Path(REPORT_PATH, "roc_curve.png"))


if __name__ == "__main__":
    
    dataset = import_data("data/bank_data.csv")
    perform_eda(dataset)

    x_train, x_test, y_train, y_test = perform_feature_engineering(dataset)
    train_models(x_train, x_test, y_train, y_test)
