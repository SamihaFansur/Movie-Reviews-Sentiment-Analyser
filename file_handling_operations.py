# -*- coding: utf-8 -*-

#Using pandas to manipulate data
import pandas as pd


""" FileHandlingOperations Class
    A class that loads and saves predictions to and from files.
    
    Author: Samiha Fansur, The University Of Sheffield, 2023
"""

class FileHandlingOperations:
    def load_prediction_from_file(filename):
        """
        Loads predictions from a given file.

        Args:
            filename (str): Filename of file to read data from.

        Returns:
            dict:  A dictionary with sentence IDs as keys and sentiments as values.
        """
        
        with open(f"{filename}") as file:
            data_frame = pd.read_csv(file, delimiter='\t') #reading file. using tab as a delimiter
        sentiment_dict = data_frame.set_index('SentenceID')['Sentiment'].to_dict() #converting to dictionary
        return sentiment_dict


    def save_predictions_to_file(filename, predictions, new_words):
        """
        Saveing predictions into a file.

        Args:
            filename (str): File name of the file to save predictions in.
            predictions (list): Predicted sentiment values list.
            new_words (list): A list of words with predictions.
        """
        # Creating a list of tuples. tuples of sentence ids and sentiments
        combined_data = [(word.sentenceID, sentiment) for word, sentiment in zip(new_words, predictions)]
        # Creating and saving the dataframe to the file
        data_frame = pd.DataFrame(combined_data, columns=['SentenceID', 'Sentiment'])
        data_frame.to_csv(f"{filename}", sep='\t', index=False) #has no index column
        
    # unused - using this for error analysis    
    def load_actual_sentiments_from_file(filename):
        """
        Loads actual sentiments from a given file.

        Args:
            filename (str): Filename of file to read data from.

        Returns:
            dict: A dictionary with sentence IDs as keys and actual sentiments as values.
        """
        with open(filename, 'r') as file:
            # Reading file using tab as a delimiter, make sure the column name matches your file structure
            data_frame = pd.read_csv(file, delimiter='\t')  # Check if the delimiter is correct

        # Adjust the column name to match the file's structure
        sentiment_dict = data_frame.set_index('SentenceId')['Sentiment'].to_dict()  # Convert to dictionary
        return sentiment_dict    
