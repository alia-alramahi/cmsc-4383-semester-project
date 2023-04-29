# How to run the code
Open 'project.ipynb' in a Google Colab window. Ensure that the packages for the imports are installed prior to runtime. Upload both JSON files into
the Colab environment and run all cells.


1.	Topic Research: Research past work related to your project
a.	Project objective: be able to determine the sentiment of an article (sarcastic or not sarcastic).

b.	Potential applications of project: Analyze sentiment of social media posts on sites such as Twitter (A novel algorithm for sarcasm detection using supervised machine learning approach, Sarcasm detection using machine learning algorithms in Twitter: A systematic review), sometimes to detect instances of cyberbullying (Machine Learning and feature engineering-based study into sarcasm and irony classification with application to cyberbullying detection). It seems that methods for sentiment analysis in general are typically applied to interpersonal online texts, which primarily exist on social media platforms. They are also sometimes used for product reviews (Sarcasm Detection with Sentiment Semantics Enhanced Multi-level Memory Network)

c.	Known challenges of building a model: Choosing how to tokenize the input (What is Tokenization in NLP). Additionally, the nuance of tone in everyday language is not easily distinguished by a machine learning model (Sentiment Analysis Challenges and How to Overcome Them). Idioms are also not easily understood by models, which makes it hard for the model to gauge the sentiment of certain texts.

d.	Types of datasets used in the past: Datasets are always text-based, as the problem is inherently one that detects sentiment in textual media. This can range, however, from things like social media posts to emails to transcripts. 

e.	Types of methods applied in the past: Tokenization of the text to produce the vocabulary, which can then be passed into models like RNN, GRU, and LSTM models (What is Tokenization in NLP)

f.	What is the state-of-the-art model (SOTA) for this problem: Transformer-based models (What is Tokenization in NLP)

g.	What metrics are used for model success: Precision, recall, F1 score, root mean squared error (RMSE), and perplexity are a few (The Most Common Evaluation Metrics in NLP) 

2.	Dataset: Address the following about your dataset
a.	Describe project dataset: The dataset consists of two files, each containing three columns: an article title (a string), a sarcasm rating (0 or 1, 0 meaning not sarcastic and 1 meaning sarcastic), and a link to the article (a string). The two datasets are combined to create a dataset with a total of 26,712 rows.
b.	Challenge of the dataset: Since the article titles aren't very long, the model has a harder time learning the dataset than it probably would with samples that had more text to indicate sarcasm.
c.	How was data collected and labelled in the set: It appears the data was manually collected by the dataset creator. The sarcastic headlines are sourced from The Onion, while the serious headlines are sourced from HuffPost.
d.	Potential biases embedded in dataset: Since the headlines were manually selected, there is human bias in the dataset. It is possible that the dataset creators chose titles that were more obviously sarcastic or not, or did not randomize the title selection as well as they could have.

3.	Data Analysis: Perform an analysis of your dataset. (Remember this must not be done on the test set).
a.	Statistics of dataset and numerical overview: The train dataset has 21,369 rows. There are a total of 210,282 words in all the headlines combined, and 32,284 unique words. There is an average of approximately 9.8 words per title (ranging from 2 to 39 words per title) and the percentage of sarcastic headlines is about 44%. 
b.	Insight of each statistic with regards to the ML task: The training dataset is large enough for the model to be able to have a significant amount of data to train on, hopefully reducing overfitting. The same is true of having a large corpus and vocabulary size. The variance in the length of titles gives the model more varied data to work with which should also reduce overfitting. The percentage of sarcastic titles to serious titles being nearly 50% gives the model enough data from both categories to learn on, again working to reduce overfitting.

4.	Data Cleaning: Address the following questions about data cleaning…
a.	What 2 data cleaning techniques are used: Removing the article link column and dropping duplicates/NA values.
b.	Explain methods and why they were applied: Dropping the article link column was necessary since it provides very little information about the sentiment of the text. This information would just provide clutter to the model and reduce the effectiveness of its training. Dropping duplicates and N/A values ensures that the inputs to the model are unique and the model is not training on values that have no meaning.

5.	Data Processing: Transform your dataset in ways that would be useful for your project objective.
a.	What 3 data transformations are used: Removing stop words, removing titles that are less than 5 words, and tokenizing words
b.	Explain methods and why they were applied: Tokenizing the words puts the words into a format that the model can read and train on. Titles that are less than five words do not have much data for the model to train on and should be removed to prevent overfitting. Same with removing stop words. These do not give much context regarding the sentiment of the text, so they are irrelevant and should not be used to train the model since leaving them in could lead to overfitting.

6.	Model Implementation: Implement ML models for your task
a.	Implement 3 significantly different models: LSTM model, logistic regression, and decision tree
b.	Explain how the models work, why they were chosen (use visuals): These are all popular NLP models, and I wanted to try a variety of different models to showcase the strengths and weaknesses of each one. LSTM models work through the headlines sequentially, processing both long-term relationships (words in the beginning of the sequence and later on) while also more highly using short-term relationships (surrounding words) to produce a prediction. Decision tree models continuously split data on features that it learns to produce a prediction at the end of the tree. Logistic regression models train on the features of the input data and, using a probability that it calculates, performs a classification of the input text. *** Find visuals
c.	Explain strengths and weaknesses of models: LSTM models can utilize previous words in short-term memory to analyze chunks of text, while also not getting thrown off by words previously in the text. However, they may not work as well on shorter texts as they have less context clues for the long-term memory. Decision trees can mark certain words as sarcastic or not through training, much the same as a linear regression model, to produce a prediction. 

7.	Model Training and Tuning: Train and tune your models as you train them
a.	Did models overfit and how addressed: No overfitting for either model
b.	Did models underfit and how addressed: No underfitting for either model
c.	Which hyperparameters were tuned and how it affected model performance:

8.	Results: Training your models, perform a final test and estimate your model performance (Give the results with tables or graphs rather than answering the questions below in words)
a.	Metrics used to measure accuracy and why it/they are good: Accuracy because this is a "low stakes problem" so neither false positives nor false negatives have a greater impact than the other, thus we look at overall model accuracy
b.	Training time of best model(s): LSTM network took 17 seconds, decision tree took less than a second, and the regression model took less than a second
c.	Memory size of best model? All are 48 bytes 
d.	What model performed the best in each of the above criteria: The decision tree models and regression models are tied for best, since they performed comparably.
e.	Images of sample model outputs:

9.	Discussion: After training, tuning, and testing your models, do a post analysis of your experiments and models you have created
a.	Was there a single model that clearly performed the best? No, they all performed about equally
b.	Why do you think the models that performed the best were the most successful?
c.	What data transformations or hyperparameter settings lead to the most successful improvements in model performance? Removing noise was ideal for the models to perform well, but none of them performed particularly well even with this.
d.	Were the above changes that lead to improvements also observed in any of research related to your project
e.	If you were to continue this project, describe more modeling, training, or advanced validation techniques
f.	Note any other interesting observations or findings from your experiments
g.	What potential problems or additional work in improving or testing your model do you foresee if you planned to deploy it to production?
h.	If you were to deploy your model in production, how would you make your models' decisions more interpretable?

10.	References: Include references to research, tutorials, or other resources you used throughout your project

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


Articles:
https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/
https://www.repustate.com/blog/sentiment-analysis-challenges-with-solutions/
https://towardsdatascience.com/the-most-common-evaluation-metrics-in-nlp-ced6a763ac8b

10-minute presentation that touches on:
•	Project motivation
•	Project background
•	Data analysis
•	Overview of data cleaning/transformation techniques used
•	Overview of model used
•	Results
•	Considerations if this were to ever be deployed
•	Any other parts of the project that you found interesting.

