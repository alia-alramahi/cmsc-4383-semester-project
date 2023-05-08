# How to run the code
Open 'project.ipynb' in a Google Colab window. Ensure that the packages for the imports are installed prior to runtime. Upload both JSON files into
the Colab environment and run all cells.


The objective of the project is to be able to determine the sentiment of an article (whether the title of the article is sarcastic or not sarcastic). One of the potential applications of the project are analyzing the sentiment of social media posts on sites such as Twitter (A novel algorithm for sarcasm detection using supervised machine learning approach, Sarcasm detection using machine learning algorithms in Twitter: A systematic review), sometimes to detect instances of cyberbullying (Machine Learning and feature engineering-based study into sarcasm and irony classification with application to cyberbullying detection). It seems that methods for sentiment analysis in general are typically applied to interpersonal online texts, which primarily exist on social media platforms. They are also sometimes used for identifying the sentiment of product reviews (Sarcasm Detection with Sentiment Semantics Enhanced Multi-level Memory Network).

One of the known challenges of the model is choosing how to tokenize the input (What is Tokenization in NLP). Additionally, the nuance of tone in everyday language is not easily distinguished by a machine learning model (Sentiment Analysis Challenges and How to Overcome Them). Idioms are also not easily understood by models, which makes it hard for the model to gauge the sentiment of certain texts. The datasets for NLP models are always text-based, as the problem is inherently one that detects sentiment in textual media. This can range, however, from things like social media posts to emails to transcripts. The types of methods that have been applied to these kinds of datasets in the past include tokenization of the text to produce the vocabulary, which can then be passed into models like RNN, GRU, and LSTM models (What is Tokenization in NLP). The state-of-the-art model (SOTA) for this problem are transformer-based models (What is Tokenization in NLP). On this particular dataset, some code published on Kaggle use BERT, which is an example of a transformer-based model. The metrics commonly used for model success in this space include recall, F1 score, root mean squared error (RMSE), and perplexity (The Most Common Evaluation Metrics in NLP).

The project dataset consists of two files, each containing three columns: an article title (a string), a sarcasm rating (0 or 1, 0 meaning not sarcastic and 1 meaning sarcastic), and a link to the article (a string). The two datasets are combined to create a dataset with a total of 26,712 rows. Since the article titles aren't very long, the model has a harder time learning the dataset than it probably would with samples that had more text to indicate sarcasm. It appears the data was manually collected by the dataset creator. The sarcastic headlines are sourced from The Onion, while the serious headlines are sourced from HuffPost. Since the headlines were manually selected, there is human bias in the dataset. It is possible that the dataset creators chose titles that were more obviously sarcastic or not or did not randomize the title selection as well as they could have.

For my analysis of the data, I split the dataset into 80% training, 10% validation, and 10% test. The training dataset has 21,369 rows. There are a total of 210,282 words in all the headlines combined, and 32,284 unique words. There is an average of approximately 9.8 words per title (ranging from 2 to 39 words per title) and the percentage of sarcastic headlines is about 44%. The training dataset is large enough for the model to be able to have a significant amount of data to train on, hopefully reducing overfitting. The same is true of having a large corpus and vocabulary size. The variance in the length of titles gives the model more varied data to work with which should also reduce overfitting. The percentage of sarcastic titles to serious titles being nearly 50% gives the model enough data from both categories to learn on, again working to reduce overfitting. Below are two plots showing the average words per title in the training set, and the distribution of sarcastic (1) and not sarcastic (0), respectively:
  
![graphs 1 and 2](https://github.com/alia-alramahi/cmsc-4383-semester-project/blob/main/graph1.jpg?raw=true)

The only data cleaning that was necessary was dropping duplicates and NA values. Dropping duplicates and N/A values ensures that the inputs to the model are unique and the model is not training on values that have no meaning. Since quotes are sometimes used to display sarcasm, I did not want punctuation to be removed from the dataset as that would lead to a loss of information. For data processing, stop words were removed from the dataset, words were tokenized, and lemmatization. Tokenizing the words puts the words into a format that the model can read and train on. Stop words do not give much context regarding the sentiment of the text, so they are irrelevant and should not be used to train the model since leaving them in could lead to overfitting. Lemmatization, which is reducing words to their base or root, allows for the model to associate different forms of the same word together, which can improve training accuracy because the model can more easily pick up on patterns when the tense, case, and number of a word is removed.
	
The three models that were implemented were an LSTM model, logistic regression model, and decision tree model. These are all popular NLP models, and I wanted to try a variety of different models to showcase the strengths and weaknesses of each one. LSTM models work through the headlines sequentially, processing both long-term relationships (words in the beginning of the sequence and later on) while also more highly using short-term relationships (surrounding words) to produce a prediction. Decision tree models continuously split data on features that it learns to produce a prediction at the end of the tree. Logistic regression models train on the features of the input data and, using a probability that it calculates, performs a classification of the input text. LSTM models can utilize previous words in short-term memory to analyze chunks of text, while also not getting thrown off by words previously in the text. However, they may not work as well on shorter texts as they have less context clues for the long-term memory. Decision trees can mark certain words as sarcastic or not through training, much the same as a linear regression model, to produce a prediction. Below is a diagram of the sample layers of an LSTM model:

![LSTM model](https://github.com/alia-alramahi/cmsc-4383-semester-project/blob/main/lstm.jpg?raw=true)

The LSTM model overfit to the training set. To address the issue, I attempted to redistribute the training, validation, and test sets (trying out different percentages), being more stringent about data cleaning, and experimenting with the layers of the LSTM network, the number of epochs, and the number of LSTM units. Despite my attempts, it did not seem to work, unfortunately. In terms of metrics, accuracy is used because this is a "low stakes problem" so neither false positives nor false negatives have a greater impact than the other, thus we look at overall model accuracy. Although neither of the linear regression or decision tree models overfit to the training data, they had poor accuracy scores for all three sets, akin to chance. The LSTM network typically took 2-3 minutes to run, the decision tree took less than a second, and the regression model took less than a second to run. The memory size of all the models was 48 bytes. In terms of performance, the decision tree performed the best, but just barely. Below are test accuracy outputs for each model:

![LSTM performance](https://github.com/alia-alramahi/cmsc-4383-semester-project/blob/main/lstmp.jpg?raw=true)

![Decision tree performance](https://github.com/alia-alramahi/cmsc-4383-semester-project/blob/main/decision_treep.jpg?raw=true) 

![Logistic regression performance](https://github.com/alia-alramahi/cmsc-4383-semester-project/blob/main/logistic_regressionp.jpg?raw=true)


I think that the logistic regression performed best perhaps due to its simplicity. Since the LSTM model was the most complex and overfit the most, it had very poor accuracy, which I would attribute to the complexity. I think the same logic of the simplicity of the logistic regression model can be applied to the decision tree model. I think removing the stop words was likely the most effective data transform along with lemmatization since they both helped to remove the noise from the dataset. In research, lemmatization and stop word removal are very common preprocessing steps, so it follows that there is a reason for performing these transformations. If I were to continue this project, I would experiment with different datasets to see if the overfitting on the LSTM model and the poor performance of the other models is due to a poor dataset or if it is something with the preprocessing or model structures. In order to deploy a better-performing NLP to production, we would need a much larger and more diverse dataset to ensure the model performance and variance within the dataset. Perhaps the dataset could be expanded to more than simply article titles to include longer texts. To make the results more interpretable, it would be helpful to list the classifications the model made for each datapoint so that it would be more obvious where the model was failing and how to improve it. 

References:

Dataset:
https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

Research articles:
https://www-sciencedirect-com.libweb.ben.edu/science/article/pii/S0925231220304689
https://www-sciencedirect-com.libweb.ben.edu/science/article/pii/S0306457321000984
https://pure.coventry.ac.uk/ws/files/30303019/Binder10.pdf
http://www.aimspress.com/article/doi/10.3934/electreng.2022021

Tutorials:
https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
https://realpython.com/python-counter/
https://www.geeksforgeeks.org/python-lemmatization-with-nltk/


Articles:
https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/
https://www.repustate.com/blog/sentiment-analysis-challenges-with-solutions/
https://towardsdatascience.com/the-most-common-evaluation-metrics-in-nlp-ced6a763ac8b

Diagrams:
https://www.oreilly.com/content/introduction-to-lstms-with-tensorflow/
