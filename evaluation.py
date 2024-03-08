# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

""" Evaluation Class
    Evaluates the Sentiment Analysis Model.
    
    Author: Samiha Fansur, The University Of Sheffield, 2023
"""

class Evaluation:
    def __init__(self, words, predicted_labels, number_of_classes):
        """
        Initialising class.

        Parameters:
        words (list): Word object list. Object has a sentenceID and sentiment labels.
        predicted_labels (dict): Dictionary that has sentenceID as the key and values as predicted sentiment labels.
        number_of_classes (int): Total sentiment classes.
        """
        self.words = words
        self.predicted_labels = predicted_labels
        self.number_of_classes = number_of_classes
        self.confusion_matrix = self.confusion_matrix_calc() #initialising confusion matrix

    def increment_matrix_value(self, matrix, row, column):
        """ 
        Increments the value of the matrix at the specified row and column.
        
        Args:
            matrix: Matrix to increment value in
            row: row in which to increment value in
            column: column in which to increment value in
        """
        matrix[row][column] += 1
    
    def confusion_matrix_calc(self):
        """
        Confusion matrix calculations using true and predicted sentiment labels.
    
        Returns:
        list: Confusion matrix as a 2D list.
        """
        number_of_classes = self.number_of_classes #initialising number of classes
    
        # Confusion matrix initialised with zeros. Dimensions set as number of classes
        confusion_matrix = [[0] * number_of_classes for _ in range(number_of_classes)]

        # For each word in the list of words
        # get the predicted sentiment label using the word's sentenceID
        # and increments the value in the matrix at the correct position using the actual sentiment and predicted sentiment
        for word in self.words:
            # Retrieve the predicted sentiment label for the current word's sentenceID.
            predicted_sentiment = self.predicted_labels.get(word.sentenceID)
            # actual sentiment - row, predicted sentiment - column
            self.increment_matrix_value(confusion_matrix, word.sentiment, predicted_sentiment)
        
        # Return confusion matrix
        return confusion_matrix

    def f1_score_calc(self, true_positives, false_positives, false_negatives):
        """
        Calculates F1 score for a given class.

        Args:
            true_positives (int): True positives count in class.
            false_positives (int): False positives count in class.
            false_negatives (int): False negatives count in class.

        Returns:
            float: The class's f1 score.
        """
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = (2 * precision * recall) / (precision + recall)
        return f1_score
    
    def macro_f1_score(self):
        """
        Calculates macro-average F1 score for all classes.
    
        Returns:
            float: Macro-average F1 score.
        """
        confusion_matrix = np.array(self.confusion_matrix) #converting to array for easier manipulation
        f1_scores = []
        
        # for each class calculate f1 score
        for i in range(confusion_matrix.shape[0]):
            true_positives = confusion_matrix[i, i] # diagonal element at [i,i]
            false_positives = confusion_matrix[i].sum() - true_positives # sum of the row for class minus tp
            false_negatives = confusion_matrix[:, i].sum() - true_positives #sum of the column for class minus tp
            # calculating f1 score for class and adding to list
            f1_scores.append(self.f1_score_calc(true_positives, false_positives, false_negatives)) 

        macro_avg_f1 = sum(f1_scores) / len(f1_scores) #averaging f1 scores for all classes
        return macro_avg_f1

    def create_heatmap(self, data, title, xlabel, ylabel):
        """
        Use data to create heatmap

        Args:
            data (list): Data, 2D list, for which to create the heatmap for.
            title (str): Heatmap title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
        """
        data_frame = pd.DataFrame(data)
        ax = sns.heatmap(data_frame, annot=True, fmt="d") #formatting annotations as integers
        ax.set_title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
    
    def plot_confusion_matrix(self):
        """
        Plotting heatmap using confisuon matrix.
        """
        self.create_heatmap(self.confusion_matrix, 'Confusion matrix', 'Predicted', 'True')
        plt.show()
