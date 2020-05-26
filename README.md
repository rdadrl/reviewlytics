# Reviewlytics
Predicts accuracy of Amazon reviews by sentiment analysis.

## Strategy
* Select sample data
	* Picked software related categories for the sake of context
	* Pareto's Principle: the law of the vital few requires proper selection of vital group.
		* Selected as a sample generalization technique.
		* Dataset offers review votes; higher vote count with high helpfulness rate can be considered more "truthful".
	* Verified Purchases
		* An Amazon verified purchase tends to be more credible.
* Form vocabulary
	* Preprocces
		* remove punctuations
		* tokenize
		* remove stop words
		* stem
	* Select most frequent N-words
	* Fill Bag of Words (BOW)
* Model data
	* Logistic Regression
	* Multinomial/Bernoulli Naive Bayesian
	* Support Vector Classifier
* Run statistical analysis' on data
	* Use model to find and filter outliers (in respect to stdev of actual score to predicted score)
	* Use model to predict overall score
* Compare findings

## Datasets
* [Amazon Review Data](https://s3.amazonaws.com/amazon-reviews-pds/readme.html) 
