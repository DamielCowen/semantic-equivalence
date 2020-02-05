import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve

class vector_comparison:
    
    '''

    NLP pipeline for comparing two inputs to eachother. Takes paired inputs, preforms preprocessing, fits model, 
    computs vectors. Returns results.
    
    INPUTS: dataframe
    
    Objective: return list of cosine similarity values
    
    '''
    
    def __init__(self, data):
        self.df = data

        
    def split_data(self,X_label, y_label):
        '''
        splits data into train and test set for model validation.
        '''
        X = self.df[X_label]
        y = self.df[y_label]
                        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        self.fitting_text = X_train['question1'].values +' '+X_train['question2']

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        return X_train, X_test
        
        
    def fit_tfidf(self, stopwords = None, ngram_range = (1,1), max_df = 1, min_df = 1, max_features = None):
        #assumes data split. Uses self.fittingtext to fit the tdif
        
        self.tfidf = TfidfVectorizer(stop_words = stopwords, ngram_range =ngram_range, max_df = max_df, 
                                     min_df = min_df, max_features = max_features)
        self.tfidf_fit = self.tfidf.fit(self.fitting_text)
        self.X_train_vector = self.tfidf_fit.transform(self.fitting_text).toarray()
        
        
    def fit_pca(self, n_components = 1):
        #input are the vectors from tfidf
        
        self.pca = PCA(n_components = n_components)
        self.pca_fit = self.pca.fit(self.X_train_vector)

        
    def transform_tfidf(self, column):
        #assumes tfidf has been fit.         
        return self.tfidf_fit.transform(column).toarray()
  

    def transform_pca(self, column):
        #assumes pca has been fit.        
        return self.pca_fit(column).toarray()
    
    
    def compute_cosine_similarity(self, arr1, arr2):
        # computes the cosine similarity between each vector in arr1 and arr2. Assumes arr1 and arr2 
        # are equal length and that each vector is of the same dimensionality.
        output = []

        for i in range(len(arr1)):
            output.append(float(cosine_similarity(arr1[i].reshape(1, -1) ,arr2[i].reshape(1, -1) )))
            
        return output
    
    
    def compute_jiccard_distance(self, arr1, arr2):
        output = []

        for i in range(len(arr1)):
            output.append(float(cosine_similarity(arr1[i].reshape(1, -1) ,arr2[i].reshape(1, -1) )))
            
        return output
        
        
    def return_df_w_results(self, results):
        #assumes test data has been transformed in a format that matches training data. Does not check this.
        self.X_test['results'] = results
        self.final = self.df.join(self.X_test['results'], how = 'inner', on = self.df.index).drop(columns ='key_0')
        
        
    def compute_confusion_matrix(self, threshold =0.7):
        #assumes that return_df_w_results has been run and self.final exsits        
        threshold = threshold
        correct = 0
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        for idx in self.final.index:
            guess = (self.final['results'].loc[idx] > threshold).astype(int)
            actual =  self.final['is_duplicate'].loc[idx]

            if guess == actual:
                correct += 1
                if guess == 1:
                    true_pos += 1
                if guess == 0:
                    true_neg += 1
            elif guess == 1 and actual == 0:
                false_pos += 1
            else:
                false_neg += 1
                
        #print(true_pos, " ", false_pos)
        #print(false_neg," ", true_neg)
        
        self.true_pos, self.false_pos, self.false_neg, self.true_neg = true_pos, false_pos, false_neg, true_neg
        self.accuracy = (true_pos + true_neg) / self.final.shape[0]
        self.TPR = true_pos / (self.final['is_duplicate'] == True).astype(int).sum()
        self.FPR = true_neg / (self.final['is_duplicate'] == False).astype(int).sum()
        self.precision = self.true_pos / (self.true_pos +self.false_pos)

    def model_report(self):
        
        self.compute_confusion_matrix()
        
        report = f''' TP:{self.true_pos}, FP: {self.false_pos}, FN:{self.false_neg}, TN: {self.true_neg} TPR: {self.TPR}, FPR: {self.FPR}, Accuracy: {self.accuracy}, Precison: {self.precision}'''
        
        print(self.true_pos, " ", self.false_pos)
        print(self.false_neg," ", self.true_neg)
        
        return report
        
        
    
    def optimize_threshold(self):
        
        threshold = list(range(0,1000))
        top_acc = 0
        top_thresh = 0
        
        for t in threshold:
            current = t/1000
            self.compute_confusion_matrix(current)
            
            if self.accuracy > top_acc:
                top_acc = self.accuracy
                top_thresh = current
         
        
        return top_acc, top_thresh
    
    def draw_roc(self):
                
        threshold = list(range(0,1000))
       
        list_thresh = []
        list_TPR = []
        list_FPR = []
        self.roc_auc_score = roc_auc_score(self.final['is_duplicate'],self.final['results'])
        self.fpr, self.tpr, self.thresholds = roc_curve(self.final['is_duplicate'], self.final['results'])
        
   
        return self.thresholds, self.tpr, self.fpr
            
        
        
            
            
           
            
            
            
        
    
    
        
        
    


        