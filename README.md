# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The data is related to individual information used for banking purpose in tabular form, with many features such as age, loan, education and job. Using this past data we want to fit a model that will predict if he/she will subscribe to the service or not. 

To solve this problem we used two machine learning features of azure ML,Hyperdrive and AutoML. We found that **voting ensemble** gave best accuracy of – 91.73% 

## Scikit-learn Pipeline
Diagram:



In Scikit-learn Pipeline, we cleaned the data and used logistic regression to classify the data. We choose parameters like – max_iteration and C (for regularization) for tuning the model.

Azure hyperdrive provides feature of hyperparameter tuning which saves significant amount of time in tuning the model. Just like in this case, I varied value of C from 0.5 – 2 (Continuous and uniformly distributed) and max_iter from 50 – 150 (Discrete data with interval of 10). 

parameter_space={
    '--C':uniform(0.5, 2),
    '--max_iter':quniform(50, 150, 10)}


I choose RandomParameterSampling sampler, as it is faster and allows early stopping.

       

Benefits of the early stopping:
Policy is used for early stopping poorly performing runs.

policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

As I used banditpolicy with slack_factor=0.1, it will terminate the run which is having less accuracy than best accuracy by factor 0.1. It helps in efficient use of computational resource. 

 

## AutoML
AutoML is great feature. We provided only data as a input and rest is all done by AutoML.

automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task="classification",
    primary_metric='accuracy',
    training_data=data,
    label_column_name='y',
    n_cross_validations=4)

First it checked quality of data (If there is any missing features or unbalance data) then it has generated 41 models and gets the best model having highest accuracy. 

Model selected by AutoML is Voting ensemble which has 91.73% accuracy.

## Pipeline comparison
There are few fundamental differences in architectures using HyperDrive and AutoMl pipelines. In first case we fix the underlying model/algorithm and fine tune the hyper parameters. Where as in AutoMl, based on the data provided, different machine learning and deep learning models are tested and best model is chosen based on the performance.

In our case AutoML outperformed logistic regression using hyperdrive. Accuracy of Voting ensamble (AutML) is 91.73% where as logistic regression tuned for best parameter gave 91.19% accuracy.

AutoML does exhaustive analysis and covers wide range of models and saves huge amount of work.   

## Future work
We can add more steps in data cleaning and feature engineering. In AutoML we didn’t add deep learning models; we can enable them also in future. In Pipeline using hyperdrive, we can build more custom and data oriented model and can do rigorous fine tuning using hyperDrive.

## Proof of cluster clean up
I deleted it in jupyter notebook.
