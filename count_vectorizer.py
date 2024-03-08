# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict

""" CountVectorizer Class
    For a given sentence it counts the instances of every unique word
    
    Author: Samiha Fansur, The University Of Sheffield, 2023
"""

class CountVectorizer:
    def collect_all_words(self, sentences):
        """
        Collects all words from a list of sentences.

        Args:
            sentences (list): A list of sentences for which to count words in
        
        Returns:
            list: A list containing all words from all sentences.
        """
        # Creating a single big list of all words(flattens all lists)
        return [word for sentence in sentences for word in sentence.tokens]

    def count_vectorizer(self, sentences):
        """
        Creates a count vectorizer for given sentences.

        Args:
            sentences (list): A list of sentences to vectorize.
        
        Returns:
            dict: A dictionary where keys are unique words and values are their corresponding
              count across all sentences.
        """
        # Collect all words from the sentences
        all_words = self.collect_all_words(sentences)
        # Create a set of unique words
        unique_words = set(all_words)
        # Counting the occurrence of each unique word in all sentences
        return {word: np.sum([word in sentence.tokens for sentence in sentences]) for word in unique_words}
