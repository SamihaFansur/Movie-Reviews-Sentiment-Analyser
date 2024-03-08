# -*- coding: utf-8 -*-

""" TrainNaiveBayes Class
    Trains the Sentiment Analysis model.

    Author: Samiha Fansur, The University Of Sheffield, 2023
"""

class TrainNaiveBayes:

    def __init__(self, words, sentiment_class_list):
        """
        Initialising class.

        Args:
            words (list): Words list from training data.
            sentiment_class_list list[int]: Sentiment class.
        """
        self.sentiment_class_list = sentiment_class_list
        self.number_of_words = len(words)
        compute_training_data = self.train(words, sentiment_class_list)

        # Setting attributes using computed training data
        self.sentence_count_in_sentiment  = compute_training_data["sentence_count_in_sentiment "]
        self.number_token_sentiments = compute_training_data["number_token_sentiments"]
        self.token_counts_in_sentiment  = compute_training_data["token_counts_in_sentiment "]
        self.magnitude_of_vocab = compute_training_data["magnitude_of_vocab"]
    
    def initialize_counters(self, sentiment_class_list):
        """
        Initializing counters for every class.

        Args:
            sentiment_class_list list[int]: Sentiment class.

        Returns:
            tuple: Tuple which contains initialised counters for number of words, token count, 
            and a list of dictionaries for relative frequencies.
        """
        word_count = [0] * len(sentiment_class_list) # count of words for each sentiment class
        token_count = [0] * len(sentiment_class_list) # total token count for each sentiment class
        # dictionary to story relative token counts for each sentiment class
        rel_token_count_dict = [dict() for _ in range(len(sentiment_class_list))]
        #creating a tuple
        return word_count, token_count, rel_token_count_dict
    
    def process_sentiment_tokens(self, sentiment, tokens, word_count, token_count, rel_token_count_dict):
        """
        Update counters after processing tokens for each sentiment class.
    
        Args:
            sentiment (int): Sentiment class of current tokens.
            tokens (list): List of tokens for processing.
            word_count (list): Number of words in every class.
            token_count (list): Token count in every sentiment class.
            rel_token_count_dict (list of dicts): Relative frequency of each token for every sentiment class.
        """
        word_count[sentiment] += 1 # Increment word count for a given sentiment
        token_count[sentiment] += len(tokens) # Increment token count for a given sentiment
    
        # For each token in the set of token, count the number of occurrences
        # of each token in the given sentiment
        for token in tokens:
            # If the token exists in dictionary, increment its count, else add it with a count of 1.
            rel_token_count_dict[sentiment].setdefault(token, 0)
            rel_token_count_dict[sentiment][token] += 1
            
    def calculate_vocab_size(self, words):
        """
        Calculates vocabulary size in dataset.
    
        Args:
            words (list): Words list from training data.
    
        Returns:
            int: Vocabulary size.
        """
        # For each token for every word in words list, add token to set. Using set to avoid duplicates
        unique_tokens = set(token for word in words for token in word.tokens)
        return len(unique_tokens) # vocabulary size

    def train(self, words, sentiment_scale):
        """
        Giving the model words and number of classes to train it.

        Args:
            words (list): Words list from training data.
            sentiment_scale (int): Count of sentiment classes from dataset.

        Returns:
            dict: A dictionary that stores computed training data.
        """
        # Initialize variables for each sentiment class
        word_count, token_count, rel_token_count_dict = self.initialize_counters(sentiment_scale)
    
        # Getting sentiment and tokens for every word
        sentiment_tokens = [(word.sentiment, word.tokens) for word in words]
    
        # Process each sentiment-token pair
        for sentiment, tokens in sentiment_tokens:
            self.process_sentiment_tokens(sentiment, tokens, word_count, token_count, rel_token_count_dict)
    
        # Calculating the number of unique tokens (vocabulary)
        magnitude_of_vocab = self.calculate_vocab_size(words)
    
        # Return all the combined computed training data as a dictionary
        return {
            "sentence_count_in_sentiment ": word_count,
            "number_token_sentiments": token_count,
            "token_counts_in_sentiment ": rel_token_count_dict,
            "magnitude_of_vocab": magnitude_of_vocab,
        }