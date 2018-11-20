import pandas as pd
import numpy as np



from sklearn.base import BaseEstimator, TransformerMixin
class TransformationWrapper(BaseEstimator,TransformerMixin):
    
    def __init__(self,fitation= None, transformation = None): 
        
        self.transformation = transformation
        self.fitation = fitation
        
    
        
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.data_ = None
        self.column_name_ = X.columns[0]
        if self.fitation == None:
            return self
        
        self.data_ = [self.fitation(X[self.column_name_])]
        return self  
    
    def transform(self, X, y=None): 
        X = pd.DataFrame(X)
        
        if self.data_ != None:
            return pd.DataFrame(X.apply(
                lambda row: pd.Series( self.transformation(row[self.column_name_], self.data_)),
                axis = 1
            ))
        else:
            return pd.DataFrame(X.apply(
                lambda row: pd.Series( self.transformation(row[self.column_name_])),
                axis = 1
            ))
        
        
from sklearn.preprocessing import LabelEncoder
class LabelEncoderP(LabelEncoder):
    def fit(self, X, y=None):
        super(LabelEncoderP, self).fit(X)
    def transform(self, X, y=None):
        return pd.DataFrame(super(LabelEncoderP, self).transform(X))
    def fit_transform(self, X, y=None):
        return super(LabelEncoderP, self).fit(X).transform(X)