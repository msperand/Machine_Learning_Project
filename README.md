![robots voting](https://raw.githubusercontent.com/msperand/Machine_Learning_Project/main/Other/DALL·E%202024-05-21%2016.19.32%20-%20A%20fun%20illustration%20depicting%20different%20machine%20learning%20models%20as%20robots%2C%20each%20voting%20according%20to%20their%20accuracy.%20Imagine%20robots%20named%20DistillBERT%2C%20C.webp)

# Machine_Learning_Project
Predicting the difficulty of French text (Kaggle competition)

## The Task
This repository is the result of the Machine Learning Project of Antoine Trabia and Maxime Sperandio. We form the team "Apple" in the Kaggle competition. The main goal of this project was to predict the difficulty of French sentences. We were given 4'800 French sentences and their linked difficulty in the CEFR format (A1, A2, B1, B2, C1, C2), in the CSV: `training_data.csv`, we were also given `unlabeled_test_data.csv` which was a dataset of 1'200 French sentences of which we had to predict the difficulty. After having done that, we had to upload our predictions on Kaggle in the same format as sample_submission.csv. We will now discuss the different methods we used to try to have the best accuracy.

We will go through our main steps, detailing what we did, why, and with what result. 
Firstly, we proudly arrived first in the aforementioned competition, with a final accuracy of 65.2% which was the best result from our 130 submissions over one month.

## The Repository
We organized the repository with one folder with the different notebooks linked to the methods we used, the data that we received, the one we created, and then everything we needed for our Streamlit application. Finally, there is the video of presentation of our work.

## The start
What we decided to do first was to get to know the training and test data. For the `training_data`, we did some basic EDA that can be found in the notebook of classic tries, and for the `test_data`, we did six submissions with only one category each time to know as best we could what the data looked like. What we found out was that both datasets were evenly distributed and that there was a small correlation between the length of the sentences and their difficulty.

## Classic Tries
We then decided to do most of the classic algorithms we knew, giving us the following results.

|           |   Naive Bayes |   Logistic Regression |   Decision Tree |      KNN |   Random Forest |      SVM |   XGBoost |   AdaBoost |
|:----------|--------------:|----------------------:|----------------:|---------:|----------------:|---------:|----------:|-----------:|
| accuracy  |      0.410845 |              0.405631 |        0.351408 | 0.283629 |        0.384776 | 0.416058 |  0.387904 |   0.294056 |
| precision |      0.428029 |              0.411018 |        0.344254 | 0.35046  |        0.381636 | 0.417252 |  0.3815   |   0.343244 |
| recall    |      0.410845 |              0.405631 |        0.351408 | 0.283629 |        0.384776 | 0.416058 |  0.387904 |   0.294056 |
| f1        |      0.410068 |              0.404039 |        0.345034 | 0.254637 |        0.362797 | 0.415141 |  0.379298 |   0.283024 |


We believe that the reason we did not manage to get a very high accuracy, as we saw directly on the Kaggle as well, is mostly because the problem is very complex. Indeed, we are trying to classify French sentences which are a sum of different words and a lot of them.

## Neural Networks

With that in mind, we decided to try neural networks, however we did not manage to make it work. It is always overfitted and never manages to give an accuracy even close to the classical methods.

## BERT

Realizing that we turned ourselves towards the Bidirectional Encoder Representations from Transformers (BERT) models. We tried different ones, starting with `DistilBERT`, and then `Bert-multilingual` as the sentences were in French. This improved our accuracy up to 54.2% when using the `bert-base-multilingual`.

We believe that BERT models are better at predicting how hard sentences are because they understand the context well. Unlike older methods that might just look at words individually or use simple rules, BERT reads the whole sentence and knows how words relate to each other. This deep understanding helps BERT catch all the nuances and complexities, making it much more accurate at judging sentence difficulty. Moreover, BERT's advanced architecture is great at picking up on long-range connections and intricate patterns in text, which older models often miss.

## FlauBERT Large

The next big improvement was when we started using `FlauBERT-large`, indeed doing some more research and trying different things, we saw that there were BERT models already made for French that we could fine-tune on our data. We found two main ones, `CamemBERT` and `FlauBERT`. We focused mainly on FlauBERT for two reasons. Firstly, because we started with it and it allowed us to jump to more than 59% at the first use, and secondly, because we never managed to make anything out of the `CamemBERT-large`. We tried different types with different hyperparameters, but it never managed to learn and kept predicting only one class.

After some research we realized that `FlauBERT-large` works better than `CamemBERT-large` for predicting sentence difficulty because it’s specifically designed for the French language, capturing its nuances and complexities more effectively. Trained on a diverse and extensive French corpus, FlauBERT excels at understanding the subtleties of French text. While `CamemBERT` is also powerful, `FlauBERT`’s tailored training gives it an edge in handling the specific linguistic features and challenges of French, leading to more accurate predictions of sentence difficulty.

The 1024 embeddings and 374M parameters of FlauBERT-large, make it an amazing model to work with, as it is able to grasp the complexity and nuances of the sentences.

To optimize the hyperparameters, we tried different methods, like optuna and grid-search. However, those require running the training at least 5 to 10 times, which takes time even with the fastest machine of Google Colab, so we decided to do it by hand and change them ourselves. This allowed us to have a lot of different submissions, varying the number of epochs, the decay, steps, batch size, and even random seed and state.  This turned out to be very useful as we'll see later, and gave us a best accuracy of 59.5% on the Kaggle.

## Data Augmentation

Even with this, we realised that we were stagnating at that level of accuracy. Looking to improve that, we turned to data augmentation.

### Back translation
Our first try was backtranslation. It is the concept of translating text from one language to another and then translating it back to the original language to check for accuracy and consistency. It allows us to add variations to the sentences while ensuring the meaning remains intact, which is very useful for our task. We tried from French to English and then back to French. 
We first tried with chat gpt directly, but it was not able to do the whole dataset because of its size. We then went with the API from Google Translate but got blocked by the number of requests and then the time it took. We thus decided to go in another direction.

### Synonym replacement

the concept is easy on paper, replace one word with a synonym. This becomes a little bit more tricky when the goal is to keep the same difficulty. For that reason we first asked chat gpt to do the task, resulting in the dataset "augmented_training_data_chat_synonym.csv". 
As a second try, we created a big library of synonyms and a corpus of difficulty with the help of chat gpt to make the replacement by hand, creating the dataset "augmented_data" in the notebook "Project-FlauBERT".

### Paraphrasing
The goal of this method is to paraphrase the sentence to keep it close to the original one, making it of the same difficulty but adding variety in the training_data. We did it with chat gpt resulting in the dataset "enhanced_paraphrased_data.csv".

All these methods were very useful in improving our model as they allowed us to bring it up to 63.5% accuracy. However, it is not really for the reason one may think. indeed, it was mostly because it increased the number of sentences allowing the model to learn more from them, as the new datasets, consisted mostly of the same sentences repeated many times. This is because our synonym replacement had only a limited number of words, thus leaving most sentences unchanged. the same phenomenon happened with the paraphrasing, probably because Chat gpt did a poor job not actually paraphrasing. 

We realized this only late when we had already used this extensively, while improving our accuracy, for this reason, we decided to keep it that way. We believe that what happened is that our models overfitted the training data but improved as the test data is very close to it, but we will discuss it more in the limitations part.

## Mastermind

During the competition, we quickly realized that we needed a distinctive approach to stand out. The previously mentioned steps were natural progressions in the competition context, and it was likely that several groups would adopt similar strategies. This scenario would turn the competition into a race to find the best parameters, data augmentation techniques, and so on, making it challenging to maintain the lead due to some level of luck and constant pressure. Therefore, we needed a unique strategy, something other competitors wouldn't think of doing. We devised our special strategy when we were stagnating below the 60% mark, with other competitors close behind. At that point, we had made nearly 30-40 submissions, leading us to develop the "mastermind" strategy.

Inspired by the game Mastermind, where players combine different colored pegs to form sequences and refine their guesses based on feedback, we applied a similar concept to our model predictions. The goal was to combine the predictions of various models to achieve a more accurate final prediction. We employed two main strategies:

Firstly, the weighted strategy involves allowing each model to "vote" on the predicted difficulty level. These votes are weighted according to each model's accuracy, giving more influence to the more accurate models. The difficulty level with the highest weighted vote count is chosen as the final prediction.

Secondly, the outliers strategy focuses on correcting a model's predictions using the consensus of other models. If all models predict the same difficulty level except for one, the outlier model's prediction is replaced with the consensus prediction. This approach relies on the premise that if most models agree on a prediction, the deviating model is likely incorrect.

Throughout the competition, we experimented with several variants of the mastermind strategy and combinations of models. This often led to increases of 2-3% in our prediction accuracy compared to our best singular model. This improvement allowed us to comfortably surpass our competitors. Ultimately, our winning mastermind combination involved using the weighted strategy with our best four models and further correcting the resulting predictions with the next best three models. This approach increased our accuracy from 63.5% (our best individual model) to 65.2%, securing our victory in the competition.

The mastermind strategy was devised for the competition only. For practical application, we implemented the 63.5% accuracy model in our app, as we believed a difference of two percent wouldn't significantly impact performance outside of the competition.

## Other tries

We did many other tries that ranged from not very successful to absolutely not working, we are going to list them here. We already mentioned the problems encountered with back translation and the optuna library. In addition to that, we tried different models that we simply were not able to make run, for example, we tried the model Mixtral from the HuggingFace website, but it seemed to block us. We also wanted to try the OpenAI APIs to fine-tune a GPT but we kept getting denied access. Finally, we found a library of French synonyms called synonyms that we wanted to use for the synonym replacement, but it was not well developed and we could not use it.


## Limitations

Now that we have mentioned all the good things about this project, let us talk about the limitations of our models. First of all, even with a first place we are only at a 65.2% accuracy. This means that around 35% of the time we are wrong. Secondly, as we mentioned before, our BERT models tended to overfit badly, and it made them more accurate. Indeed, duplicating the sentences in the training set led to a better score. This is counterintuitive and comes probably from the fact that the training and test data are very close to each other. However, it also suggests that if we tried our model on something slightly different, it could perform very badly.

## The Streamlit
A popular way to learn a language is by listening to music in that language. Many people use this technique to learn French, but one common challenge is knowing which songs to choose. If the song is too simple, the benefits are minimal, but if the song is too complex, it can be overwhelming. To address this issue, we developed 'Lyrical Lingo' an app designed to help French learners select songs that match their proficiency level. Our app estimates the overall difficulty level of a French song (A1, A2, B1, B2, C1, C2) and assesses the difficulty of each sentence in the song. 

'Lyrical Lingo' comes with a selection of preloaded songs, ranging from French classics to contemporary hits. Additionally, the app allows users to test the difficulty of their own song choices.

As mentioned earlier, our app uses a `FlauBERT-large` model trained on augmented data. This model achieved an accuracy of 63.5% in predicting test data on Kaggle. The model is available on Huggingface (URL: [FlauBERT French Song Difficulty Model](https://huggingface.co/AntoineTrabia/FrenchSongDifficulty/tree/main)). 

Here is the link for the app: (URL: [Lyrical Lingo](https://lyrical-lingo.streamlit.app)). The python file 'LyricalLingo.py' for the app is also available on the GitHub in the 'Streamlit' directory and can be run locally. 

## The video
The video explaining our project is available on YouTube. (URL: [Youtube Video](https://www.youtube.com/watch?v=uQUQ9VL6Zug))

