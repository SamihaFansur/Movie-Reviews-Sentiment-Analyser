# -*- coding: utf-8 -*-

""" SentimentLexicon Class
    Sentiment lexicon approach using Bing Liu Model for feature extraction.

    Author: Samiha Fansur, The University Of Sheffield, 2023
"""

class SentimentLexicon:
    def __init__(self, positive_lexicon_path, negative_lexicon_path):
        """
        Initialising class.

        Args:
            positive_lexicon_path (str): The file path to the positive word lexicon.
            negative_lexicon_path (str): The file path to the negative word lexicon.
        """
        self.positive_words = self.load_lexicon(positive_lexicon_path)
        self.negative_words = self.load_lexicon(negative_lexicon_path)

    def load_lexicon(self, file_path):
        """
        Loads words from a given lexicon file into a set.

        Args:
            file_path (str): The file path to the lexicon file.

        Returns:
            set: A set of words from the lexicon file.
        """
        with open(file_path, 'r') as file:
            # removes any leading/trailing whitespace
            words = set(word.strip() for word in file.readlines())
        return words

    def get_sentiment_score(self, word):
        """
        Returns a sentiment score for a given word.

        Args:
            word (str): The word for which to calculate the sentiment score.

        Returns:
            int: Sentiment score of the word.
        """
        if word in self.positive_words:
            return 1
        elif word in self.negative_words:
            return -1
        else:
            return 0 # neutral words
