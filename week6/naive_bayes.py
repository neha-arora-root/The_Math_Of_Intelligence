import numpy as np
import pandas as pd
import math
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class NaiveBayes(object):
    
    def __init__(self, categories, fileName, tfidf=True):
        try:
            col_names = ['category', 'message']
            messages = pd.read_csv(fileName, header=None, names=col_names, encoding="utf-8")
            total_examples = messages.shape[0]
            self.training_examples = int(math.floor(0.7*total_examples))
            self.test_examples = int(total_examples - self.training_examples)
            print ("Total examples: "+str(total_examples))
            print ("Training set size: "+str(self.training_examples))
            print ("Test set size: "+str(self.test_examples))
            indices = list(range(total_examples))
            random.shuffle(indices)
            self.train_messages = messages.ix[indices[0:self.training_examples],:]
            self.test_messages = messages.ix[indices[self.training_examples:total_examples],:]
        except OSError as err:
            print("OS error: {0}".format(err))
        self.categories = self.create_categories(categories)
        self.words = defaultdict(dict)
        self.words_message_count = defaultdict(dict)
        self.unique_words = set()
        if tfidf:
            self.tfidf = True
        
    def train_data(self):
        mul = self.training_examples//10
        counter = 0
        for idx, message in self.train_messages.iterrows():
            self.train(message)
            if (counter+1)%mul == 0:
                print ("Training done for {}/{} examples".format(counter+1, self.training_examples))
            counter += 1
        print("Done!")
    
    def create_categories(self, categories):
        message_categories = {category: {'total': 0, 'word_count': 0} 
                      for category in categories}
        return message_categories
    
    def train(self, message):
        category = str(message['category'])
        
        # Process message and get a list of useful tokens
        words = self.process_message(message['message'])
        
        # Find the unique words in the list of words and combine it with unique_words set
        self.unique_words = set(list(self.unique_words) + words)
        
        # for a given word, update the count of category in which it is seen
        for word in words:
            if self.words[word].get(category):
                self.words[word][category] += 1
            else:
                self.words[word][category] = 1
            
        # Update the count of the times this category message has been seen
        self.categories[category]['total'] += 1
        
        # Update the count of words seen for a particular category
        self.categories[category]['word_count'] += len(message)
        
        if self.tfidf:
            # Find inverse document frequency for each word
            unique_message_words = set(words)
            for word in unique_message_words:
                if self.words_message_count[word].get(category):
                    self.words_message_count[word][category] += 1
                else:
                    self.words_message_count[word][category] = 1
                
                if self.words_message_count[word].get('total_count'):
                    self.words_message_count[word]['total_count'] += 1
                else:
                    self.words_message_count[word]['total_count'] = 1
        
    
    def classify(self, message):
        scores = {}
        best_fit = {}
        best_score = float("-inf")
        sum_scores = 0.0
        words = self.process_message(message)
        for category, category_data in self.categories.items():
            prior_log_prob = self.get_category_log_prob(category, category_data['total'])
            term_frequency_log_prob = self.get_predictors_log_likelihood(category, words)
            inv_doc_frequency_log_prob = 0.0
            if self.tfidf:
                inv_doc_frequency_log_prob = self.get_predictors_log_idf(category, words)
            predictors_log_likelihood = term_frequency_log_prob + inv_doc_frequency_log_prob
            
            score = math.exp(prior_log_prob + predictors_log_likelihood)
            scores[category] = score
            sum_scores += score
            if score > best_score:
                best_fit["category"] = category
                best_fit["score"] = score
                best_score = score
        
        for score_key in scores.keys():
            scores[score_key] /= sum_scores
        best_fit['score'] = scores[best_fit['category']]
        return (best_fit, scores)
    
    def get_category_log_prob(self, category, count):
        prob = float(count)/(self.training_examples + len(self.categories.keys()))
        return math.log(prob)
    
    def get_predictors_log_idf(self, category, words):
        log_prob = 0.0
        for word in words:
            smoothed_frequency = self.training_examples
            if self.words_message_count.get(word) and self.words_message_count[word].get(category):
                smoothed_frequency = self.words_message_count[word]['total_count']
            idf = math.log(self.training_examples/smoothed_frequency)
            log_prob += idf
        if log_prob == 0.0:
            return 0.0
        return math.log(log_prob)
    
    def get_predictors_log_likelihood(self, category, words):
        word_count = self.categories[category]['word_count'] + len(self.unique_words)
        log_likelihood = 0
        for word in words:
            smoothed_frequency = 1
            if (self.words.get(word) and self.words[word].get(category)):
                smoothed_frequency += self.words[word][category]
            log_likelihood += math.log(smoothed_frequency/float(word_count))
        return log_likelihood
    
    def process_message(self, message, lower_case=True, stem=True, stop_words=True, gram=1):
        if lower_case:
            message = message.lower()
        words = word_tokenize(message)
        words = [w for w in words if len(w) > 2]
        if gram > 1:
            w = []
            for i in range(len(words)-gram+1):
                w += ' '.join(words[i:i+gram])
            return w
        if stop_words:
            sw = stopwords.words('english')
            words = [word for word in words if word not in sw]
        if stem:
            stemmer = PorterStemmer()
            words = [stemmer.stem(word) for word in words]
        return words
    
    def test(self):
        accuracy = 0.0
        '''
        size = len(self.categories)
        confusion_matrix = pd.DataFrame(np.zeros((size, size)))
        category_names = list(self.categories.keys())
        confusion_matrix.columns = category_names
        
        for idx, message in self.test_messages.iterrows():
            best_fit, _ = self.classify(message['message'])
            actual_category = int(message['category'])
            pred_category = best_fit['category']
            confusion_matrix.iloc[actual_category][pred_category] += 1
        '''
        #accuracy = np.trace(confusion_matrix)/self.test_examples
        #return {'confusion_matrix':confusion_matrix, 'accuracy': accuracy}
        for idx, message in self.test_messages.iterrows():
            best_fit, _ = self.classify(message['message'])
            #print ("\n"+message['message'])
            #print ("Actual: "+str(message['category'])+", Predicted: "+str(best_fit['category']))
            
            if str(best_fit['category']) == str(message['category']):
                accuracy += 1
        accuracy /= self.test_examples
        return {'accuracy': accuracy}
        