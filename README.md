# Text Classification: Toxic Comment Analysis

## Introduction

Conversation AI team has developed several models to improve the monitoring of conversation online. The main issue of these models (despite the error in the prediction as such) is the fact that they don’t allow to label the type of toxicity of the comments analyzed. This means that we cannot really generalize the use of these predictions, as not all online platforms may be willing to act on a text (e.g., with censorship) with the same level of intervention. It will mainly depend on the internal policy of the company. Therefore, the possibility to develop a model that will grant the possibility to define and label the type/tone of a text data type into a specific category, e.g., “insult”, “threats”, etc., it is quite interesting. 

In order to do so, we will be using a set of NLP models, with increasing complexity, to identify the one that is better performing with the provided dataset and possibly scalable on other NLP text classification problems. The models we have identified for this project are: “Bag of Words, “FastText”, "Glove" - “Bidirectional Long Short-Term Memory (LSTM)”.  
The benchmark model that will be used is the simpler model at the base of Text Classification, i.e. “Bag of Words”. In this model, a text is represented as a “bag” (multiset) of words, disregarding grammar and even word order but keeping multiplicity. We will be using a sklearn implementation of “Bag of Words”, TfidfVectorizer, to convert the provided text into a numerical matrix that can be than mapped on the pre-defined set of label provided (i.e. “toxic”, “severe toxic”, “obscene”, “threat”, “insult” and “identity hate”). We will then use a Logistic Regression, Random Forest an Naive Bayes algorithms on data created by the TfidfVectorizer, and will keep as benchmark the one which is better performing. The prediction will be then realizes by using a Multi-Output Classifier from sklearn to predict all 6 categories. 

The application of NLP and Text Classification which will be used in this project is referring to a Kaggle competition held in 2017-2018, the [“Toxic Comment Classification Challenge”](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

The project proposal submitted with the previous project is [here](https://review.udacity.com/#!/reviews/3203822)

## Installation

There are three main Jupyther Notebbok that have been used: 

1. Toxic_Comment_Analysis_Bag_of_Words: where we have implemented and evaluated the “Bag of Words model;
2. Toxic_Comment_Analysis_FastText: where we have implemented and evaluated the FastText model;
3. Toxic_Comment_Analysis_Glove: where we have imported the Glove embeddings and implemented/evaluated a bidirectional LSTM model

## Library

The main library used are:

- pandas
- numpy
- matplotlib
- seaborn
- statistics
- string
- sklearn.model_selection
- sklearn.feature_extraction
- sklearn.feature_extraction.text
- sklearn.multioutput
- sklearn.naive_bayes
- nltk.tokenize
- nltk.corpus
- string
- tqdm.notebook
- sklearn.multioutput 
- fasttext
- sklearn.metrics
- keras.preprocessing.text
- keras.preprocessing.sequence
- tensorflow.keras.utils
- keras.models
- keras.layers
- tensorflow.keras.utils
- keras.layers

## Evaluations

In order to evaluate the performance of our models, we will use the AUC-ROC Curve. We have decided to use this evaluation metrics as the best one in terms of multi-class classification problem. AUC stands for “Area under the Curve”, while ROC stands for “Receiver Operating Characteristics”. The AUC-ROC metric clearly helps determine and tell us about the capability of a model in distinguishing the classes. The judging criteria being – Higher the AUC, better the model. AUC-ROC curves are frequently used to depict in a graphical way the connection and trade-off between sensitivity and specificity for every possible cut-off for a test being performed or a combination of tests being performed.
