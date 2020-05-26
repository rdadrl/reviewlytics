import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.text import Text
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
from nltk import pos_tag
from nltk import FreqDist
import string, re
from tqdm import tqdm

class SentimentAnalyzer:
	# args:
	#	n_words			: amount of features to be considered
	#	verified_only	: take only verified purchases into consideration
	#	pareto			: take only 20% of most helpful reviews into account.
	def __init__(self, data, n_words = 5000, verified_only = False, pareto = False):
		print("Running Sentiment Analysis")
		self.data = []
		if pareto:
			print("Applying Pareto's Law into dataset.")
			#Pareto strategy:
			#Pick 20% of data of most helpful reviews equally distributed from each label group
			star_dict = {"1": [], "2": [], "3": [], "4": [], "5": []} 

			sorted = data.sort_values(by="helpful_votes", ascending = False)
			for index, row in tqdm(sorted.iterrows()):
				star_dict[str(row.star_rating)].append(row)
			

			target_sample_length = round(len(data.review_id) / 5 / 5)
			min_sample_length = min(len(star_dict["1"]), len(star_dict["2"]), len(star_dict["3"]), len(star_dict["4"]), len(star_dict["5"]))
			print("Requiring", target_sample_length, "samples per star category.")
			print("Found", min_sample_length, "samples least per star category.")

			target_sample_length = min(min_sample_length, target_sample_length)
			print("Picking", target_sample_length, "samples per category.")
			
			for i in range(0, target_sample_length):
				self.data.append(star_dict["1"][i])# + int((len(star_dict["1"]) - target_sample_length) / 2))
				self.data.append(star_dict["2"][i])# + int((len(star_dict["2"]) - target_sample_length) / 2))
				self.data.append(star_dict["3"][i])# + int((len(star_dict["3"]) - target_sample_length) / 2))
				self.data.append(star_dict["4"][i])# + int((len(star_dict["4"]) - target_sample_length) / 2))
				self.data.append(star_dict["5"][i])# + int((len(star_dict["5"]) - target_sample_length) / 2))

			self.data = pd.DataFrame(self.data)
		else:
			print("No Pareto's.")
			self.data = data
			print(self.data)

		self.data.reset_index(inplace=True)
		print(self.data.helpful_votes.mean())
		self.vocabulary = []
		self.verified_only = verified_only
		self.pareto = pareto
		self.stop_words = list(set(stopwords.words('english')))
		self.allowed_word_types = ["J"] #  j is adject, r is adverb, and v is verb
		self.stemmer = SnowballStemmer('english')
		self.lemmatizer = WordNetLemmatizer()

		print("Preprocessing text...\t\t[1/3]")
		self.preprocess()

		print("Generating global features...\t[2/3]")
		word_list = FreqDist(self.vocabulary)
		self.features = list(word_list.keys())[:n_words]
		
		print("Generating bag of words...\t[3/3]")
		self.bow = self.generate_bag_of_words()
		print("Task done")

	def preprocess(self):
		for i in tqdm(range(0, len(self.data.review_body))):
			#there were some random nan values- this seperates them.
			review = self.data.review_body[i]
			if (review != review):
				i = i + 1

			#puncts_removed = re.sub(r'[^(a-zA-Z)\s]','', review)
			#tokenized = word_tokenize(puncts_removed)
			#tagged = word_tokenize(review)
			#stopped = [w for w in tokenized if not w in self.stop_words]

			#select adjects
			#result = [] #clean, preprocessed tokens
			#tagged = pos_tag(stopped)

			#for j in range(0, len(tagged)):
				#w = tagged[j][1][0]
			#	result.append(tagged[j])#self.stemmer.stem(tagged[j][0].lower()))

				#if w in self.allowed_word_types:
			#	self.vocabulary.append(result[j])

			puncts_removed = re.sub(r'[^(a-zA-Z)\s]','', review)
			tokenized = word_tokenize(puncts_removed)
			stopped = [w for w in tokenized if not w in self.stop_words]

			#select adjects
			result = [] #clean, preprocessed tokens
			tagged = pos_tag(stopped)

			for j in range(0, len(tagged)):
				w = tagged[j][1][0]
				result.append(self.lemmatizer.lemmatize(tagged[j][0].lower()))

				if w in self.allowed_word_types:
					self.vocabulary.append(result[j])
			self.data.review_body[i] = result

	def generate_bag_of_words(self):
		result = []
		for i in tqdm(range(0, len(self.data.review_body))):
			if (not self.verified_only) or (self.verified_only and self.data.verified_purchase[i] == "Y"):
				review = self.data.review_body[i]
				score = self.data.star_rating[i]
				label = "neg"
				if score > 3:
					label = "pos"
				features = {}

				for f in self.features:
					features[f] = (f in review)

				result.append((features, score))

		return result	

	def get_data(self):
		return self.data
