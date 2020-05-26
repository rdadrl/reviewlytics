import pandas as pd
from sentimentAI.SentimentAnalyzer import *
import random

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, accuracy_score

#data = pd.read_csv("./resources/data/sample_us.tsv", error_bad_lines=False, sep='\t')
data = pd.read_csv("./resources/data/amazon_reviews_us_Digital_Software_v1_00.tsv.gz", compression='gzip', error_bad_lines=False, sep='\t')
print("done reading training data", len(data.review_body))

sa = SentimentAnalyzer(data.sample(n=10000), pareto = True, verified_only = True)
training_set = sa.bow

validation_sample = data.sample(n=1000)
validation_reviews = validation_sample.review_body.tolist()
sa = SentimentAnalyzer(validation_sample)
validation_set = sa.bow 
ground_truth = [r[1] for r in validation_set]

#print("Training MultinomialNB")
#MNB_clf = SklearnClassifier(MultinomialNB())
#MNB_clf.train(training_set)
#MNB_pred = [MNB_clf.classify(r[0]) for r in validation_set]
##for i in range(len(MNB_pred)):
##	if MNB_pred[i] == 5:
##		print(validation_sample.review_body[i])
##		print(validation_reviews[i],"\n+++++++++++++++++++++++++++++++")
#print("Got F1 score of", precision_score(ground_truth, MNB_pred, average='micro'))
#
#print("Training BernoulliNB")
#BNB_clf = SklearnClassifier(BernoulliNB())
#BNB_clf.train(training_set)
#BNB_pred = [BNB_clf.classify(r[0]) for r in validation_set]
#print("Got F1 score of", precision_score(ground_truth, BNB_pred, average='micro'))

print("Training LogisticRegression")
LogReg_clf = SklearnClassifier(LogisticRegression())
LogReg_clf.train(training_set)
LogReg_pred = [LogReg_clf.classify(r[0]) for r in validation_set]
print("Got F1 score of", precision_score(ground_truth, LogReg_pred, average='micro'))

#print("Training SGD")
#SGD_clf = SklearnClassifier(SGDClassifier())
#SGD_clf.train(training_set)
#SGD_pred = [SGD_clf.classify(r[0]) for r in validation_set]
#print("Got F1 score of", precision_score(ground_truth, SGD_pred, average='micro'))#