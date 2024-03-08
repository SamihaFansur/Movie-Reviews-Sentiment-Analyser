# -*- coding: utf-8 -*-

#Using pandas to manipulate data
import pandas as pd

""" Sentence Class
    Represent a sentence from a movie review dataser that has tokens and a sentiment.
    
    Author: Samiha Fansur, The University Of Sheffield, 2023
"""

class Sentence:
    def __init__(self, sentenceID, tokens, sentiment):
        """
        Sentence Object initialisation
        
        Args:            
            sentenceID (int): unique id for a sentence.
            tokens (list of str): Tokenized form of the sentence.
            sentiment (int): the sentiment label for a sentence.
        """
        self.sentenceID = sentenceID
        self.tokens = tokens
        self.sentiment = sentiment

    def wordList(filename):
        """
        Creates a list of Sentence objects after reading in a file.
        Uses 'Phrase' column to get tokens and the 'Sentiment' column for 
        the sentiment labels.

        Parameters:
            filename (str): File path.

        Returns:
            list: A list of Sentence objects.
        """
        
        sentenceObjs = []
        data_frame = pd.read_csv(filename, index_col=0, delimiter='\t') #Reading file into dataframe
        sentiment_present = 'Sentiment' in data_frame.columns #checking if sentiment column exists in data frame

        # For each row in the dataframe 
        # if sentiment is present create a Sentence object with corresponding
        # sentiment otherwise set sentiment to -1
        for index, row in data_frame.iterrows():
            if sentiment_present:
                sentence = Sentence(index, row['Phrase'].split(), row['Sentiment'])
            else:
                sentence = Sentence(index, row['Phrase'].split(), -1)
            sentenceObjs.append(sentence)

        return sentenceObjs