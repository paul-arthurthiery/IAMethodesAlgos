from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class SoftmaxClassifier(BaseEstimator, ClassifierMixin):  
    """A softmax classifier"""

    def __init__(self,omega = 1, lr = 0.1, alpha = 100, n_epochs = 1000, eps = 1.0e-5,threshold = 1.0e-10 , regularization = True, early_stopping = True):
       
        """
            self.lr : the learning rate for weights update during gradient descent
            self.alpha: the regularization coefficient 
            self.n_epochs: the number of iterations
            self.eps: the threshold to keep probabilities in range [self.eps;1.-self.eps]
            self.regularization: Enables the regularization, help to prevent overfitting
            self.threshold: Used for early stopping, if the difference between losses during 
                            two consecutive epochs is lower than self.threshold, then we stop the algorithm
            self.early_stopping: enables early stopping to prevent overfitting
        """

        self.lr = lr 
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.eps = eps
        self.regularization = regularization
        self.threshold = threshold
        self.early_stopping = early_stopping
        self.omega = omega
        

    """
        Public methods, can be called by the user
        To create a custom estimator in sklearn, we need to define the following methods:
        * fit
        * predict
        * predict_proba
        * fit_predict        
        * score
    """


    """
        In:
        X : the set of examples of shape nb_example * self.nb_features
        y: the target classes of shape nb_example *  1

        Do:
        Initialize model parameters: self.theta_
        Create X_bias i.e. add a column of 1. to X , for the bias term
        For each epoch
            compute the probabilities
            compute the loss
            compute the gradient
            update the weights
            store the loss
        Test for early stopping

        Out:
        self, in sklearn the fit method returns the object itself


    """


    def fit(self, X, y=None):
        
        prev_loss = np.inf
        self.losses_ = []

        self.nb_feature = X.shape[1]
        self.nb_classes = len(np.unique(y))

        

        #X_bias = np.ones(np.size(X),np.size(X[0])+1)
        #X_bias[:,:-1] = X 
        
        X_bias = np.c_[X , np.ones(len(X))]
        
        self.theta_ = np.random.rand(self.nb_feature, self.nb_classes)
        self.theta_ = np.vstack([self.theta_ ,self.theta_[1]])

        for epoch in range(self.n_epochs):
            logits = np.dot(X_bias, self.theta_)
            probabilities = self.predict_proba(X)
            loss = self._cost_function(probabilities, y)               
            self.theta_ -= self.omega*self._get_gradient(X_bias,y,probabilities)    
            self.losses_.append(loss)
            if(epoch > 0):
                if((self.losses_[-2] -self.losses_[-1]) <= self.threshold):
                    self.early_stopping = True
                else:
                    self.early_stopping = False
                    if self.early_stopping:
                        pass
                               
        print("Done")
        return self    

    """
        In: 
        X without bias

        Do:
        Add bias term to X
        Compute the logits for X
        Compute the probabilities using softmax

        Out:
        Predicted probabilities
    """

    def predict_proba(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        X_bias = np.ones((np.shape(X)[0],np.shape(X)[1]+1))
        X_bias[:,:-1] = X
        result = np.ones((np.shape(X)[0],self.nb_classes))
        for i in range(np.shape(X)[0]):  
            z = np.dot(X_bias[i,:], self.theta_)
            result[i,:] = self._softmax(z)
        return result    


        """
        In: 
        X without bias

        Do:
        Add bias term to X
        Compute the logits for X
        Compute the probabilities using softmax
        Predict the classes

        Out:
        Predicted classes
    """
    
    def predict(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        X_bias = np.ones((np.shape(X)[0],np.shape(X)[1]+1))
        X_bias[:,:-1] = X
        
        proba = self.predict_proba(X)
        result = np.zeros((np.shape(X)[0],1))
        for i in range(np.shape(X)[0]):
            prob = proba[i,:]
            maxProb = np.max(prob)
            for j in range(self.nb_classes):
                if(prob[j] == maxProb):
                    result[i] = j
        return result

    

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X,y)


    """
        In : 
        X set of examples (without bias term)
        y the true labels

        Do:
            predict probabilities for X
            Compute the log loss without the regularization term

        Out:
        log loss between prediction and true labels

    """    

    def score(self, X, y=None):
        probs = self.predict(X)
        self.regularization = False
        result = self._cost_function(probs,y)
        #print(result)
        return result
        

    """
        Private methods, their names begin with an underscore
    """

    """
        In :
        y without one hot encoding
        probabilities computed with softmax

        Do:
        One-hot encode y
        Ensure that probabilities are not equal to either 0. or 1. using self.eps
        Compute log_loss
        If self.regularization, compute l2 regularization term
        Ensure that probabilities are not equal to either 0. or 1. using self.eps

        Out:
        Probabilities
    """
    
    def _cost_function(self,probabilities, y ):
        probabilities = np.clip(probabilities, self.eps, 1-self.eps)
        y=self._one_hot(y)
        result = 0
        for i in range(np.shape(probabilities)[0]):
            classePredite = np.where(probabilities[i] == max(probabilities[i]))
            classeCorrecte =  np.where(y[i] == 1)
            if(classeCorrecte == classePredite):
                result += np.log10(probabilities[i][classeCorrecte])
        result *= (-1/(np.shape(probabilities)[0]))
        if(self.regularization):
            return result
            somme = 0
            for i in range(self.nb_feature):
                for k in range(self.nb_classes):
                    somme += self.theta_[i][k]**2
            result += somme*self.alpha
        else:
            return result
                
        
    """
        In :
        Target y: nb_examples * 1

        Do:
        One hot-encode y
        [1,1,2,3,1] --> [[1,0,0],
                         [1,0,0],
                         [0,1,0],
                         [0,0,1],
                         [1,0,0]]
        Out:
        y one-hot encoded
    """

    
    
    def _one_hot(self, Y):
        size = np.size(Y)
        unique = np.unique(Y)
        min = unique[0]
        sizeUnique = np.size(unique)
        
        result = np.zeros((size,sizeUnique))
        
        for i in range(size):
                result[i][Y[i] - min] = 1
        return result
    """
        In :
        Logits: (self.nb_features +1) * self.nb_classes

        Do:
        Compute softmax on logits

        Out:
        Probabilities
    """
    
    def _softmax(self,z):
        result = np.zeros((1, self.nb_classes))
        expz = np.sum(np.exp(z))
        for k in range(self.nb_classes):
            result[:,k] = np.exp(z[k])/expz
        return result
    """
        In:
        X with bias
        y without one hot encoding
        probabilities resulting of the softmax step

        Do:
        One-hot encode y
        Compute gradients
        If self.regularization add l2 regularization term

        Out:
        Gradient

    """
    def _get_gradient(self,X,y, probas):
      X = X.T
      y=self._one_hot(y);
      result = np.dot(X,(probas - y))
      result *= 1/(np.size(probas))
      if(self.regularization):
          return result
          somme = 0
          for i in range(self.nb_feature):
              for k in range(self.nb_classes - 1):
                  somme += self.theta_[i][k]**2
                  result += somme*self.alpha
      else:
          return result
                

