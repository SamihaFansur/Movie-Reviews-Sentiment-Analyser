# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np


""" TfIdf Class
    Calculating Tf-Idf for feature extraction.
    Code is adapted from assignment 1 to be used for this assignment.
    
    Author: Samiha Fansur, The University Of Sheffield, 2023
"""

class TfIdf:
    """
    
    Attributes:
    documents (list of str): The list of documents for TF-IDF computation.
    word_set (set of str): A set of unique words across all documents.
    document_word_matrix (list of dict): A list of dictionaries where each dictionary
                                         represents the term frequency in a document.
    idf_values (dict): A dictionary storing the inverse document frequency for each word.
    """

    def __init__(self, documents):
        """
        Initialising TfIdf class.

        Parameters:
        documents (list of str): Document list for TF-IDF calculation.
        """
        self.documents = [str(doc) for doc in documents]  # Making sure all documents are strings
        self.word_set = set(word for doc in self.documents for word in doc.split())  # Picking unique words
        self.document_word_matrix = self.create_document_word_matrix()  # Creating the Tf(term frequency) matrix
        self.idf_values = self.compute_idf_values()  # Calculting IDF values for every word

    def create_document_word_matrix(self):
        """
        Creating a matrix of term frequency for each document.

        Returns:
        list: List of dictionaries for the term frequency in a document.
        """
        document_word_matrix = []
        for doc in self.documents:
            doc_dict = defaultdict(int)  # Default dictionary to store word frequency
            for word in doc.split():  # Splitting the document into words
                doc_dict[word] += 1  # Incrementing the count for each word
            document_word_matrix.append(doc_dict)
        return document_word_matrix

    def compute_idf_values(self):
        """
        Calculating Idf values for each word.

        Returns:
        dict: A dictionary with words and keys and IDF scores as values.
        """
        idf_values = {}
        total_documents = len(self.documents)  # Total number of documents
        for word in self.word_set:
            word_in_docs_count = sum(word in doc.split() for doc in self.documents)
            # Calculating IDF with an offset to ensure non-zero values
            idf_values[word] = np.log(total_documents / float(word_in_docs_count)) + 0.89
        return idf_values

    def compute_tf_idf(self):
        """
        Calculating TF-IDF for every word in every document.

        Returns:
        list: List of dictionaries for the Tf-Idf scores for a document.
        """
        tf_idf_matrix = []
        for doc_dict in self.document_word_matrix:
            doc_tf_idf = {}  # Dictionary to store TF-IDF for each word in a document
            for word, tf in doc_dict.items():
                # Multiplying term frequency (TF) with the IDF value for the word
                doc_tf_idf[word] = tf * self.idf_values[word]
            tf_idf_matrix.append(doc_tf_idf)
        return tf_idf_matrix
