# -*- coding: utf-8 -*-

from collections import defaultdict

""" SentimentBias Class
    Calculare sentiment bias for both 3 and 5 classes.
    
    Author: Samiha Fansur, The University Of Sheffield, 2023
"""

class SentimentBias:
    def initialize_weights(self, number_of_classes):
        """
        Initialising class.

        Args:
            number_of_classes (int): Number of sentiment classes, 3 or 5.

        Returns:
            list: Weights list corresponding to the sentiment class.
        """
        # sentiment weights based on number of classes.
        return [-1, 0, 1] if number_of_classes == 3 else [-2, -1, 0, 1, 2]
    
    def calc_sentiment_bias(self, token_sentiment_count, number_of_classes):
        """
        Sentiment bias calculation for a token using its sentiment distribution.

        Args:
            token_sentiment_count (list): Number of sentiments for a token as a list.
            number_of_classes (int): Number of classes.

        Returns:
            int: Sentiment bias.
        """
        weights = self.initialize_weights(number_of_classes) #initilaise weights
        # Calculate sentiment bias
        return sum(token_sentiment_count[i] * weights[i] for i in range(number_of_classes)) # weighted sum of sentiments

    def sort_biases_descending(self, biases):
        """
        Sort the biases in descending order.

        Args:
            biases (list): Tuples list. Tuple contains a token and its bias.

        Returns:
            list: List of sorted biases.
        """
        return sorted(biases, key=lambda item: item[1] * item[1], reverse=True) #sorting biases
    
    def reduced_sorted_biases(self, biases, select_feature_percentage):
        """
        Picking a top percentage of features based on biases.

        Args:
            biases (list): Tuples list. Tuple contains a token and its bias.
            select_feature_percentage (float): Percentage of features to select.

        Returns:
            list: List of biases selected.
        """
        sorted_biases = self.sort_biases_descending(biases) #sort biases
        number_of_features_to_keep = int(len(biases) * select_feature_percentage)
        return sorted_biases[:number_of_features_to_keep] #top percentage of features

    def count_token_sentiments(self, words, number_of_classes):
        """
        Distribution count of sentiments for every token.

        Args:
            words (list): Words list with tokens and sentiment.
            number_of_classes (int): Number of classes.

        Returns:
            defaultdict: Dictionary with tokens as keys and sentiment counts as values.
        """
        token_sentiment_count = defaultdict(lambda: [0] * number_of_classes) #initialise dictionary
        # for every word in list, count the number of sentiments for every token in a word
        for word in words:
            for token in word.tokens:
                token_sentiment_count[token][word.sentiment] += 1
        return token_sentiment_count
    
    def calc_all_sentiment_biases(self, token_sentiment_count, number_of_classes):
        """
        Compute the sentiment biases for all tokens.

        Args:
            token_sentiment_count (dict): Dictionary with token sentiment counts.
            number_of_classes (int): Number of classes.

        Returns:
            list: List of tuples with tokens and their biases.
        """
        # calculating sentiment bias for every token
        return [(token, self.calc_sentiment_bias(sentiment_count, number_of_classes)) 
                for token, sentiment_count in token_sentiment_count.items()]

    def filter_words_by_features(self, words, features):
        """
        Filter tokens in words that are in features.

        Args:
            words (list): Words list with tokens.
            features (set): Features to keep as a set to avoid duplicates.

        Returns:
            list: Words list with selected features.
        """
        filtered_words = []
        # for every word in words only keep tokens that are in the features set
        for word in words:
            filtered_tokens = [token for token in word.tokens if token in features]
            if filtered_tokens:
                word.tokens = filtered_tokens
                filtered_words.append(word)
        return filtered_words
