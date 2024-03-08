# -*- coding: utf-8 -*-

from math import prod

""" MultinomialNaiveBayes Class
    Uses Multinomial naive bayes classifier to predict sentiments.
    Model adapted to only work with Sentiment Bias feature extraction.
    
    Author: Samiha Fansur, The University Of Sheffield, 2023
"""

class MultinomialNaiveBayes:
    def __init__(self, data):
        """
        Naive bayes object initialisation. 
        Model is adapted so that it words for Sentiment bias method only.

        Args:
            data: An object containing the data to predict sentiments for.
        """
        self.data = data
    
    def calc_prior_probability(self, num_of_words_for_sentiment):
        """
        Calculates the prior probability of a sentiment.

        Args:
            num_of_words_for_sentiment (int): The number of words for a sentiment.

        Returns:
            float: Calculated prior probability of a sentiment.
        """
        # Dividing the number of words in a sentiment by the total number of words in the dataset.
        return num_of_words_for_sentiment / self.data.number_of_words

    def calc_all_prior_probabilities(self):
        """
        Calculates prior probabilities for all sentiments.

        Returns:
            list: List of prior probabilities for each sentiment.
        """
        return [self.calc_prior_probability(x) for x in self.data.sentence_count_in_sentiment ]

    def calc_smooth_sentiment(self, index_of_sentiment):
        """
        Laplace smoothing applied to a sentiment.

        Args:
            index_of_sentiment (int): The index of the sentiment.

        Returns:
            int: Total count of smoothed tokens for a sentiment.
        """
        sentiment_number_of_tokens = self.data.number_token_sentiments[index_of_sentiment]
        # Adding the total count of tokens for a sentiment and size of vocabulary
        return sentiment_number_of_tokens + self.data.magnitude_of_vocab

    def calc_relative_likelihood(self, token, index_of_sentiment):
        """
        For a sentiment, function calculates the relative likelihood of a token.

        Args:
            token (str): Token for which to calculate relative likelihood.
            index_of_sentiment (int): Index of sentiment.

        Returns:
            float: Relative likelihood of the token for a sentiment.
        """
        # Counting how many times a token appears in the sentiment
        rel_token_number = self.data.token_counts_in_sentiment [index_of_sentiment].get(token, 0)
        smooth_sentiment = self.calc_smooth_sentiment(index_of_sentiment) # Smoothed total count for sentiment
        return (rel_token_number + 1) / smooth_sentiment #Applying laplace smoothing
    
    def calc_all_relative_likelihoods(self, word, index_of_sentiment):
        """
        Calculates relative likelihoods for all tokens of a word in a sentiment.

        Args:
            word: The word containing the tokens.
            index_of_sentiment (int): Index of sentiment.

        Returns:
            list: List of relative likelihoods for every token in the word.
        """
        # Returns zero if number of words for sentiment is zero
        if self.data.sentence_count_in_sentiment [index_of_sentiment] == 0:
            return 0
        # Calculating relative likelihoods for each token in the word for a sentiment.
        return [self.calc_relative_likelihood(token, index_of_sentiment) for token in word.tokens]

    def calc_likelihood(self, word, index_of_sentiment, prior_probability):
        """
        Calculates the overall likelihood of a word in a sentiment.
    
        Args:
            word: The word that contains tokens.
            index_of_sentiment (int): Index of sentiment.
            prior_probability (float): Prior probability of sentiment.
    
        Returns:
            float: Overall likelihood of the word in a sentiment.
        """
        # All relative likelihoods for each token in the word
        calc_all_relative_likelihoods = self.calc_all_relative_likelihoods(word, index_of_sentiment)
        
        # Overall likelihood of the word in sentiment.
        return prior_probability * prod(calc_all_relative_likelihoods)
        
    def calc_all_likelihoods(self, word):
        """
        Calculates the likelihoods of a word for all sentiments.
    
        Args:
            word: The word that contains tokens.
    
        Returns:
            list: List of likelihoods for all sentiments.
        """
        calc_all_prior_probabilities = self.calc_all_prior_probabilities() #prior probability calculation for all sentiments
        # Calculating the likelihood for the word across each sentiment
        return [self.calc_likelihood(word, i, calc_all_prior_probabilities[i]) for i in range(len(self.data.sentiment_class_list))]
     
    def predict_sentiment(self, word):
        """
        Predicting the sentiment of a word.
    
        Args:
            word: The word that contains tokens.
    
        Returns:
            int: Predicted sentiment index.
        """
        likely = self.calc_all_likelihoods(word) #Likelihoods of the word for each sentiment
        # Returning the index of the sentiment with maximum likelihood
        return max(range(len(likely)), key=lambda x: likely[x])
        
    def sentiments(self, words):
        """
        Predicting sentiments of a list of words.
    
        Args:
            words: List of word objects that have tokens.
    
        Returns:
            list: Predicted sentiments list for every word.
        """
        return [self.predict_sentiment(word) for word in words] #predict sentiment for every word in list
