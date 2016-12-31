from __future__ import division

import math
import os

import numpy as np

from collections import defaultdict

from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()

from unidecode import unidecode

# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'

# Stopwords from http://www.ranks.nl/stopwords
stopwords = ["a", "about", "above", "after", "again", "against", "all",
    "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd",
    "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
    "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most",
    "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours	ourselves", "out", "over",
    "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]


# Path to dataset
PATH_TO_DATA = './large_movie_review_dataset'
# e.g. "/users/brendano/inlp/hw1/large_movie_review_dataset"
# or r"c:\path\to\large_movie_review_dataset", etc.
TRAIN_DIR = os.path.join(PATH_TO_DATA, "train")
TEST_DIR = os.path.join(PATH_TO_DATA, "test")

def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """

    bow = defaultdict(float)

    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)

    for token in lowered_tokens:
        bow[token] += 1.0

    return bow

def tokenize_doc_stopwords(doc):
    """
    Student Implemented

    Tokenize a document and return its bag-of-words representation neglecting stopwords.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc neglecting stopwords.
    """

    global stopwords

    bow = defaultdict(float)

    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)

    for token in lowered_tokens:
        if token not in stopwords:
            bow[token] += 1.0
    return bow

def tokenize_doc_stopwords_custom(doc):
    """
    Student Implemented

    Tokenize a document and return its bag-of-words representation neglecting stopwords and custom stopwords.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc neglecting stopwords and custom stopwords.
    """

    global stopwords

    # Custop stopwords are words found common to both positive and negative labels in corpus
    custom_stopwords = stopwords + ["/><br", "movie", "one", "even", "like"]

    bow = defaultdict(float)

    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)

    for token in lowered_tokens:
        if token not in custom_stopwords:
            bow[token] += 1.0
    return bow

def tokenize_doc_stopwords_and_stemming(doc):
    """
    Student Implemented

    Tokenize a document and return its bag-of-words representation stemming each word and neglecting stopwords and custom stopwords.
    doc - a string representing a document.
    returns a dictionary mapping each stem to the number of times it appears in doc neglecting stopwords and custom stopwords.
    """

    global stopwords

    # Custop stopwords are words found common to both positive and negative labels in corpus
    custom_stopwords = map(lambda t: st.stem(t), stopwords + ["/><br", "movie", "one", "even", "like"])

    bow = defaultdict(float)

    tokens = unidecode(unicode(doc, 'utf-8')).split()
    lowered_tokens = map(lambda t: st.stem(t.lower()), tokens)

    for token in lowered_tokens:
        if token not in custom_stopwords:
            bow[token] += 1.0
    return bow

class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self, feature_extractor=tokenize_doc):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()

        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float) }

        # A function to map strings into bag-of-words models
        self.feature_extractor = feature_extractor

    def train_model(self, num_docs=None):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.

        num_docs: set this to e.g. 10 to train on only 10 docs from each category.
        """

        if num_docs is not None:
            print "Limiting to only %s docs per clas" % num_docs

        pos_path = os.path.join(TRAIN_DIR, POS_LABEL)
        neg_path = os.path.join(TRAIN_DIR, NEG_LABEL)
        print "Starting training with paths %s and %s" % (pos_path, neg_path)
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            filenames = os.listdir(p)
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print "REPORTING CORPUS STATISTICS"
        print "NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL]
        print "NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL]
        print "NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL]
        print "NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL]
        print "VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab)

    def update_model(self, bow, label):
        """
        Student Implemented

        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words/stems to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each class (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each class (self.class_total_doc_counts)
        """
        for word in bow:
            if word not in self.class_word_counts[label]:
                self.class_word_counts[label][word] = 0
            self.class_word_counts[label][word] += bow[word]
        self.class_total_word_counts[label] += sum(bow.values())
        self.vocab = self.vocab.union(set(bow.keys()))
        self.class_total_doc_counts[label] += 1.0
        pass


    def tokenize_and_update_model(self, doc, label):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not

        Make sure when tokenizing to lower case all of the tokens!
        """

        bow = self.feature_extractor(doc)
        self.update_model(bow, label)

    """
    Functions to get metrics about the corpus and model.
    """

    def top_n(self, label, n):
        """
        Returns the most frequent n tokens for documents with class 'label'.
        """
        return sorted(self.class_word_counts[label].items(), key=lambda (w,c): -c)[:n]

    def p_word_given_label(self, word, label):
        """
        Student Implemented

        Returns the probability of word given label (i.e., P(word|label))
        according to this NB model.
        """
        return self.class_word_counts[label][word] / self.class_total_word_counts[label]

    def p_word_given_label_and_psuedocount(self, word, label, alpha):
        """
        Student Implemented

        Returns the probability of word given label with psuedo counts.
        alpha - psuedocount parameter
        """
        return (self.class_word_counts[label][word] + alpha) / (self.class_total_word_counts[label] + (len(self.vocab) * alpha))

    def log_likelihood(self, bow, label, alpha):
        """
        Student Implemented

        Computes the log likelihood of a set of words give a label and psuedocount.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; psuedocount parameter
        """
        prob = 0
        for word in bow:
            prob += math.log(self.p_word_given_label_and_psuedocount(word, label, alpha)) * bow[word]
        return prob

    def log_prior(self, label):
        """
        Student Implemented

        Returns a float representing the fraction of training documents
        that are of class 'label'.
        """
        return math.log(self.class_total_doc_counts[label] / sum(self.class_total_doc_counts.values()))

    def unnormalized_log_posterior(self, bow, label, alpha):
        """
        Student Implemented

        alpha - psuedocount parameter
        bow - a bag of words (i.e., a tokenized document)
        Computes the unnormalized log posterior (of doc being of class 'label').
        """
        return self.log_likelihood(bow, label, alpha) + self.log_prior(label)

    def classify(self, bow, alpha):
        """
        Student Implemented

        alpha - psuedocount parameter.
        bow - a bag of words (i.e., a tokenized document)

        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior).
        """
        pos = self.unnormalized_log_posterior(bow, POS_LABEL, alpha)
        neg = self.unnormalized_log_posterior(bow, NEG_LABEL, alpha)
        if pos > neg:
            return POS_LABEL
        else:
            return NEG_LABEL

    def likelihood_ratio(self, word, alpha):
        """
        Student Implemented

        alpha - psuedocount parameter.
        Returns the ratio of P(word|pos) to P(word|neg).
        """
        return self.p_word_given_label_and_psuedocount(word, POS_LABEL, alpha) / self.p_word_given_label_and_psuedocount(word, NEG_LABEL, alpha)

    def evaluate_classifier_accuracy(self, alpha):
        """
        alpha - psuedocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """

        correct = 0
        total = 0
        pos_path = os.path.join(TEST_DIR, POS_LABEL)
        neg_path = os.path.join(TEST_DIR, NEG_LABEL)
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            filenames = os.listdir(p)
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    bow = tokenize_doc_stopwords_custom(content)
                    if self.classify(bow, alpha) == label:
                        correct += 1
                    total +=1
        return correct / total

def plot_psuedocount_vs_accuracy(psuedocounts, accuracies):
    """
    A function to plot psuedocounts vs. accuries. You may want to modify this function
    to enhance your plot.
    """

    import matplotlib.pyplot as plt

    plt.plot(psuedocounts, accuracies)
    plt.xlabel('Psuedocount Parameter')
    plt.ylabel('Accuracy (%)')
    plt.title('Psuedocount Parameter vs. Accuracy Experiment')
    plt.show()

if __name__ == '__main__':
    nb = NaiveBayes(tokenize_doc_stopwords_and_stemming)

    # Fully train model
    nb.train_model()

    # Evaluate approach
    accuracies = [nb.evaluate_classifier_accuracy(i) for i in range(1, 26)]
    print "Best pseudocount: " + str(np.argmax(accuracies) + 1)
    print "Best accuracy: " + str(max(accuracies))
    plot_psuedocount_vs_accuracy(range(1, 26), accuracies)
