import numpy as np

class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        # Do not change the class outside of this block
        N, D = x.shape
        if N % 2 == 1:
            median = np.sort(x,axis = 0)[(N-1)//2]
        else:
            median = np.sort(x,axis = 0)[[N//2,N//2 - 1]].mean(axis=0)
        return np.abs(x - median).mean(axis=0)# Your code here
        ####

    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        N = features.shape[0]
        self.loc = np.sort(features,axis = 0)[(N-1)//2] if N % 2 == 1 else np.sort(features,axis = 0)[[N//2,N//2 - 1]].mean(axis=0)# YOUR CODE HERE
        self.scale = np.abs(features - self.loc).mean(axis=0)# YOUR CODE HERE
        ####


    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        return -np.log(2*self.scale) - np.absolute(values-self.loc)/self.scale
        ####
        
    
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(value))
