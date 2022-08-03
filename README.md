# Social Monitoring Through Sentiment Classification And Topic Modeling For Food Delivery Platforms

Companies spend a lot of money on maintaining their brand names on social media, but they often lack the tools to systematically analyze, monitor, as well as predict how their brand images are being perceived by users. The current method (i.e focus groups) comes with an inherent time lag, and in events such as a PR crisis happening on social media, the company would not be able to react in a timely manner. 

Our project leverages supervised & unsupervised learning to build an ML application that can predict customer sentiments, understand what topics customers are talking about, and possibly detect any large swings in sentiment (e.g. PR crisis). The company domain that we will be focusing on is food delivery.

Our project is broken down into 5 steps: collecting raw data, curating our own labeled dataset, training and selecting among various classification algorithms for emotion classification, applying unsupervised learning algorithms for topic modeling, and last but not least, creating a pipeline to automate the entire process from data collection to the final dashboard.

## Please see our blog for a walkthrough of this project. Link:

## Requirements
Please see requirements.txt for requirements. In most cases, the installation of Anaconda is enough to run all the the notebooks. 


## Explanation of Notebooks
* ScrapingTool.ipynb is used to scrape twitter data 
* LabelingTool.ipynb and LabelingToolWithSplit.ipynb are our custom tool used to label the data in batches 
* labeled_csv_merging.ipynb is used to merge all of the csv files into one dataframe
* Preprocessing.ipynb is used to lowercase, lemmatize, remove stop words, etc.
* NaiveBayes_LogisticRegression_KNN_MLP.ipynb is used to test sklearn classifiers
* catboost_classifier.ipynb, decision_tree.ipynb, random_forest.ipynb, sgd_classifer.ipynb and svc_classifier.ipynb are also used to test sklearn classifiers
* bert_fine_tine.ipynb is used for the BERT model (Deleted)
* LDA.ipynb, Latent_Semantic_Indexing.ipynb, Non-negative_Matrix_Factorization.ipynb are used to test the topic modeling algorithms 

## Citations