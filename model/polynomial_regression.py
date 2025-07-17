import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from sklearn.preprocessing import StandardScaler
from io import BytesIO

image_files = []

class Polynomial_Regression:

  def __init__(self,degree=2,learning_rate=0.01,no_of_iterations=1000):
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations
    self.degree = degree

  def fit(self,X_train,Y_train):
    self.orginalX_train = X_train.copy()
    self.featureColCount = X_train.shape[1]
    self.X_train = np.hstack([X_train**i for i in range(1,self.degree+1)])
    self.Y_train = Y_train
    self.m,self.n = self.X_train.shape
    self.bias = 0
    self.weights = np.zeros(self.n) # no of weights = no of feautre columns
    
    #updating our weights using gradient decent
    for i in range(self.no_of_iterations):

      if i<50 or i%10 == 0: # only taking 1 frame out of 10
        if self.featureColCount == 1: #2d visualization
      
          x = self.X_train
          y = self.Y_train
          m = self.weights
          c = self.bias
          y_pred =  np.dot(x,m)  + c

          x = self.orginalX_train #for ploting we need original dataset
          x1 = 0 if x[0] > 0 else x[0]
          x2 = x[-1]
          y1 = 0 if y[0] > 0 else y[0]
          y2 = y[-1]

          plt.plot(x,y,"r.")
          plt.plot(x,y_pred)
          plt.xlim(x1, x2)   #x and y limts to define plot dimensions
          plt.ylim(y1, y2)
          buffer = BytesIO()
          plt.savefig(buffer, format='png')
          buffer.seek(0)
          image_files.append(imageio.imread(buffer))
          plt.clf()        #to clear plt after saving so that it won't interfere with ones in upcoming iterations 
        
        if self.n == 2: #3d visualization
          pass # to vinayak

        else:
          pass # some code to let user know "we can't make plots on 4d yet"... 
      

      self.update_weights()

  def update_weights(self):
    
    Y_predict = self.predict(self.X_train)

    #refer derivation notes for how to compute dw and db eqations
    dw = (1/self.m)*self.X_train.T.dot(Y_predict-self.Y_train) 
    db = (1/self.m)*np.sum(Y_predict-self.Y_train)

    #adjusting weights using gradient decent
    self.weights = self.weights - self.learning_rate*dw
    self.bias = self.bias - self.learning_rate*db

  def predict(self,X_test):
    return X_test.dot(self.weights) + self.bias
  
  def get_image_list(self):
    return image_files
  
