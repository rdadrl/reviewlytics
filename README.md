# Reviewlytics
Predicts accuracy of Amazon reviews by sentiment analysis.

## Strategy
* Select sample data
	* Picked software related categories for the sake of context
	* There exists digital & physical versions of same category; possibility to use one for training and the other for validation.
* Form vocabulary
	* Preproccess by stemming words
	* Select most frequent N-words
	* Fill Bag of Words (BOW)
* Model data
	* Classification: Logistic Regression using CountVectorizer
	* Clustering: Nonnegative Matrix Factorization with Term Frequency - Inverse Document Frequency.
* Run statistical analysis' on data
	* Use model to find and filter outliers
	* Use model to predict overall score
* Compare findings

## Datasets

* [Amazon Review Data](https://s3.amazonaws.com/amazon-reviews-pds/readme.html) 
