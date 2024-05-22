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

## Neural Networks

With that in mind, we decided to try neural networks, however we did not manage to make it work. It always overfitted and never manage to give an accuracy even close to the classical methods.

## BERT

Realising that we turned ourselves towards the BERT models. We tried different ones, starting with DistilBERT, and then Bert-multilingual as the sentences were in French. This improved our accuracy up to 54.2% when using the bert-base-multilingual.
We believe that BERT models are better at predicting how hard sentences are because they understand the context really well. Unlike older methods that might just look at words individually or use simple rules, BERT reads the whole sentence and knows how words relate to each other. This deep understanding helps BERT catch all the nuances and complexities, making it much more accurate at judging sentence difficulty. Moreover, BERT's advanced architecture is great at picking up on long-range connections and intricate patterns in text, which older models often miss.

## FlauBERT Large

The next big amelioration was when we started using FlauBERT-large, indeed doing some more research and trying different things, we saw that there was BERT models already made for French that we could fine tune on our own data. We found two main ones, CamemBERT and FlauBERT. We focused mainly on FlauBERT for two reasons. Firstly, because we started with it and it allowed us to jump to more than 59% at the first use, and secondly, because we never managed to make anything out of the CamemBERT-large. We tried different types with different hyperparameters, but it never managed to learn ad kept predicting only one class.
After some research we realised thatFlauBERT large works better than CamemBERT large for predicting sentence difficulty because it’s specifically designed for the French language, capturing its nuances and complexities more effectively. Trained on a diverse and extensive French corpus, FlauBERT excels at understanding the subtleties of French text. While CamemBERT is also powerful, FlauBERT’s tailored training gives it an edge in handling the specific linguistic features and challenges of French, leading to more accurate predictions of sentence difficulty.
The 1024 embeddings and 374M parameters of FlauBERT-large, make it an amazing model to work with, as it is really able to grasp the complexity and nuances of the sentences.

## Data Augmentation

### Backtranslation

### Synonym replacement

### Paraphrasing

## Mastermind

## Other tries

## Limitations



