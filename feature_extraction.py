# -*- coding: utf-8 -*-

from collections import defaultdict
from count_vectorizer import *
from sentiment_lexicon import *
from sentiment_bias import *
from tfidf import *

""" FeatureExtraction Class
    Extracts features.
    
    Author: Samiha Fansur, The University Of Sheffield, 2023
"""

class FeatureExtraction:
    def __init__(self, words):
        """
        Initialising class.
        
        Args:
            words (list): Words list from which to extract features.
        """
        positive_lexicon_path = "lexicon/positive-words.txt"
        negative_lexicon_path = "lexicon/negative-words.txt"
       
        self.features = words
        self.count_vectorizer = CountVectorizer()
        self.sentiment_lexicon = SentimentLexicon(positive_lexicon_path, negative_lexicon_path)
        self.sentiment_bias = SentimentBias()
        self.tfidf_scores = {}

    #Unused method
    def vectorize_sentences(self, sentences):
        """
        Vectorizes the given list of sentences.

        Args:
            sentences (list): Sentence list to vectorize.
        
        Returns:
            list: Sentences in a vectorized form.
        """
        return self.count_vectorizer.count_vectorizer(sentences)
    
    #Unused method
    def get_sentiment_score(self, word):
        """
        Gets the sentiment score for a word.

        Args:
            word (str): Word for which to get the sentiment score.
        
        Returns:
            int: The words' sentiment score.
        """
        return self.sentiment_lexicon.get_sentiment_score(word)
    
    #Unused method
    def get_tfidf_score(self, token):
        """
        Gets the TF-IDF score for a token.

        Args:
            token (str): Token for which to get the TF-IDF score.
        
        Returns:
            float: The tokens' TF-IDF score.
        """        
        return self.tfidf_scores.get(token, 0)
     
    #Unused method 
    def calc_tfidf_scores(self, words):
        """
        Calculating TF-IDF scores for all tokens in given sentences.
 
        Args:
            words (list): The list of sentences with tokens.
        """
        # calculating TF-IDF scores for all tokens
        all_tokens = [token for sentence in words for token in sentence.tokens]
        tfidf_vectorizer = TfIdf(all_tokens)
        tfidf_matrix = tfidf_vectorizer.compute_tf_idf()
        # Flattening the TF-IDF matrix
        self.tfidf_scores = {word: score for doc in tfidf_matrix for word, score in doc.items()}

    def featureExtraction(self, words, number_of_classes):
        """
        Extracts feeatures from words
 
        Args:
            words (list): Words list from which to extract features.
            number_of_classes (int): Number of sentiment classes, 3 or 5.
        
        Returns:
            list: Extracted features list.
        """
        # Setting the percentage of features to select based on class number
        select_feature_percentage = 0.97 if number_of_classes == 3 else 0.38
        # Calculating sentiment biases
        token_sentiment_count = self.sentiment_bias.count_token_sentiments(words, number_of_classes)
        biases = self.sentiment_bias.calc_all_sentiment_biases(token_sentiment_count, number_of_classes)
        # reducing sentiment biases to the selected percentage
        reducedBiases = self.sentiment_bias.reduced_sorted_biases(biases, select_feature_percentage)
        self.features = [token for token, _ in reducedBiases]
        filtered_words = self.sentiment_bias.filter_words_by_features(words, self.features) # extracted features
        
        
        # # Count vectorization (Not using this, as evaluation scores didn't improve and this increased runtime)
        # word_counts = self.count_vectorizer.count_vectorizer(words)
        # words_to_keep = [word for word, count in word_counts.items() if count > 4 and count < 10]
        # self.features = words_to_keep
        # return words_to_keep
        
        
        # Sentiment Lexicon (Not using this, as evaluation scores didn't improve and this increased runtime)
        # word_sentiments = [(word, self.get_sentiment_score(word)) for word in words] 
        # return word_sentiments
        

        # # TF-IDF (Not using this, as evaluation scores didn't improve and this increased runtime)
        # self.calc_tfidf_scores(words) # TF-IDF scores for tokens in the sentences
        # sentence_scores = {} # dictionary for storing total TF-IDF score for every sentence
        # # for every sentence calculate tf-idf scores
        # for sentence in words:
        #     score = sum([self.get_tfidf_score(token) for token in sentence.tokens])
        #     sentence_scores[sentence] = score
        # sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True) # sorting based on scores
        # return sorted_sentences
        
        return filtered_words
    
    