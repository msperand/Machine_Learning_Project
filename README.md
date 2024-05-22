# Machine_Learning_Project
Predicting the difficulty of French text (Kaggle competition)

## The Task
This repository is the result the Machine Learning Project of Antoine Trabia and myself, Maxime Sperandio. We form team "Apple" in the Kaggle competition. The main goal of this project was to predict the difficulty of French sentences. We were given 4'800 French sentences and their linked difficulty in the CEFR format (A1, A2, B1, B2, C1, C2), in the csv: training_data.csv, we were also given unlabeled_test_data.csv which was a dataset of 1'200 French sentences of which we had to predict the difficulty. After having done that, we had to upload our predictions on Kaggle in the same format as sample_submission.csv. We will now discuss the different method we used in order to try to have the best accuracy.

We will go through our main steps, detailing what we did, why and with what result. 
Firstly, we proudly arrived first in the aforementioned competition, with a final accuracy of 65.2% that was the best result from our 130 submissions over the span of one month.

## The start
What we decided to do first was to get to know the training and test data. For the training data, we did some basic EDA that can be found on the notebook of classic tries, and for the test data, we did six submissions with only one category each time to know as best we could what the data looked like. What we found out was that both datasets were evenly distributed and that there was a small correlation between the length of the sentences and their difficulty.

## Classic Tries
We have then decided to do most of the classic algorithm we saw, giving us the following results.

|           |   Naive Bayes |   Logistic Regression |   Decision Tree |      KNN |   Random Forest |      SVM |   XGBoost |   AdaBoost |
|:----------|--------------:|----------------------:|----------------:|---------:|----------------:|---------:|----------:|-----------:|
| accuracy  |      0.410845 |              0.405631 |        0.351408 | 0.283629 |        0.384776 | 0.416058 |  0.387904 |   0.294056 |
| precision |      0.428029 |              0.411018 |        0.344254 | 0.35046  |        0.381636 | 0.417252 |  0.3815   |   0.343244 |
| recall    |      0.410845 |              0.405631 |        0.351408 | 0.283629 |        0.384776 | 0.416058 |  0.387904 |   0.294056 |
| f1        |      0.410068 |              0.404039 |        0.345034 | 0.254637 |        0.362797 | 0.415141 |  0.379298 |   0.283024 |


We believe that the reason we do not manage to get a very high accuracy, as we saw directly on the Kaggle as well, is mostly because the problem is very complex. Indeed, we are trying to classify french sentences which are a sum of different words and a lot of them.

## BERT


