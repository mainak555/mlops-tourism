
import numpy as np

def num_features_selector(X):
    return X.select_dtypes(include=np.number).columns.to_list()   
def cat_features_selector(X):
    return X.select_dtypes(include=['object', 'category']).columns.to_list() 
