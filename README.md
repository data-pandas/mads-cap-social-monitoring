# Social Monitoring Through Sentiment Classification And Topic Modeling For Food Delivery Platforms

Companies spend a lot of money on maintaining their brand names on social media, but they often lack the tools to systematically analyze, monitor, as well as predict how their brand images are being perceived by users. The current method (i.e focus groups) comes with an inherent time lag, and in events such as a PR crisis happening on social media, the company would not be able to react in a timely manner. 

Our project leverages supervised & unsupervised learning to build an ML application that can predict customer sentiments, understand what topics customers are talking about, and possibly detect any large swings in sentiment (e.g. PR crisis). The company domain that we will be focusing on is food delivery.

Our project is broken down into 5 steps: collecting raw data, curating our own labeled dataset, training and selecting among various classification algorithms for emotion classification, applying unsupervised learning algorithms for topic modeling, and last but not least, creating a pipeline to automate the entire process from data collection to the final dashboard.

[Click here to read the full paper.](https://data-pandas.github.io/)  
[Click here to view the Live Social Monitoring Dashboard.](https://datastudio.google.com/reporting/c8b5e9f2-4760-4417-a958-e069552b9f69)  

## Requirements
Please see the `requirements.txt` for the exact version of the packages. Installing Anaconda installs most of the packages. All remaining packages can be installed using pip. 

The following packages are built into anaconda: Ipython, jupyter_client, jupyter_core, jupyterlab, matplotlib, notebook, numpy, pandas, python, scipy, seaborn, sklearn. 

The following packages are NOT built into sklearn and would need to be installed using pip: catboost, datasets, gensim, joblib, nltk, pysentimento, sentence_transformers, snscrape, torch, tranformers, tqdm. 

## Explanation of Notebooks
### Inside the Scraping, Labeling, Preprocessing, Cleaning folder
* ScrapingTool.ipynb is used to scrape Twitter data 
* LabelingTool.ipynb and LabelingToolWithSplit.ipynb are our custom tools used to label the data in batches 
* labeled_csv_merging.ipynb is used to merge all of the csv files into one dataframe
* Preprocessing.ipynb is used to lowercase, lemmatize, remove stop words, etc.
### Inside the Classification folder
* bert_base_uncased.ipynb is used for the BERT model and the final chosen model for the dashboard, referring to the following webpages: [huggingface tutorial](https://huggingface.co/docs/transformers/training), [huggingface trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer) and [colab tutorial](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb)
* catboost_classifier.ipynb, decision_tree.ipynb, NaiveBayes_LogisticRegression_KNN_MLP.ipynb, random_forest.ipynb, sgd_classifer.ipynb and svc_classifier.ipynb are also used to test sklearn classifiers
### Inside the Topic Modeling folder
* LDA.ipynb and Latent_Semantic_Indexing.ipynb are used to test the topic modeling algorithms 
* Non-negative_Matrix_Factorization.ipynb is used for topic modeling and is the final chosen model for the dashboard

## Citations
* Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). What does bert look at? an analysis of bert's attention. arXiv preprint arXiv:1906.04341.  
* Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). How to fine-tune bert for text classification? China national conference on Chinese computational linguistics.
