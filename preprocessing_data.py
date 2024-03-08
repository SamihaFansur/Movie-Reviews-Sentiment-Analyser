 # -*- coding: utf-8 -*-

from nltk.corpus import stopwords  # For removing stopwords
import string  # For removing punctuation
from nltk.stem import WordNetLemmatizer, PorterStemmer  # For lemmatizing and stemming


""" PreprocessingData Class
    Preprocesses data for sentiment analysis
    
    Author: Samiha Fansur, The University Of Sheffield, 2023
"""

class PreprocessingData:
    def __init__(self, sentences):
        """
        Initialising class with a list of sentences
        """
        self.sentences = sentences
        self.stemmer = PorterStemmer() #stemming
        self.lemmatizer = WordNetLemmatizer() #lemmatization
        # combining stopwords and punctuations into a list
        self.stopwords = set(['\'s', '``', '\'\'', '...', '--', 'n\'t', '\'d'] +
                             stopwords.words('english') + list(string.punctuation))

        # Map to scale sentiment scores  down to a 3-value scale.
        self.scale_class = [0, 0, 1, 2, 2]

    def tokens_to_lowercase(self, tokens):
        """
        Makes all words(tokens) is list lowercase.

        Parameters:
            tokens (list): A list of words(tokens).

        Returns:
            list: Lowercased tokens in a list.
        """
        return [token.lower() for token in tokens]

    def filter_stopwords(self, tokens):
        """
        Removes stopwords from a list of tokens.

        Parameters:
            tokens (list): A list of tokens.

        Returns:
            list: List of tokens after stopwords are removed.
        """
        return [token for token in tokens if token not in self.stopwords]

    def apply_stemming(self, tokens):
        """
        Stemms to a list of tokens.

        Parameters:
            tokens (list): A list of tokens.

        Returns:
            list: A list of stemmed tokens.
        """
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize_words(self, tokens):
        """
        Lemmatizes a list of tokens.

        Parameters:
            tokens (list): A list of tokens.

        Returns:
            list: List of lemmatized tokens.
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess_words(self):
        """
        Preprocesses each sentence by lowercasing, removing stopwords
        and applying stemming and lemmatization.
        
        Returns:
            list: Preprocessed sentences in a list.
        """
        for sentence in self.sentences:
            sentence.tokens = self.tokens_to_lowercase(sentence.tokens) # lowercasing tokens
            sentence.tokens = self.filter_stopwords(sentence.tokens) # Remove stopwords
            # Not applying lemmatization and stemming because scores dont't increase
            # sentence.tokens = self.apply_stemming(sentence.tokens)  # stemming
            # sentence.tokens = self.lemmatize_words(sentence.tokens) # lemmatization
        return self.sentences

  
    def scale_down_sentiment(self):
        """
        Scales down sentiment scores for every sentence.

        Returns:
            list: Scaled down sentiment scores in a list.
        """
        for word in self.sentences:
            word.sentiment = self.scale_class[word.sentiment]
        return self.sentences
