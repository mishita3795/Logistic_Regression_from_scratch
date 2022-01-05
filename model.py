# -*- coding: utf-8 -*-
"""ML Assignment 2- Implemenation of logistic regression
Authors: Krutika Hampannavar(21230039) and Mishita Kesarwani(21239405)

"""

"""Min - Max scaler function is defined below where we take 3 inputs : maximum and minimum value of a column and value which needs to be normalized.
Normalized value after scaling will be within the range of 0 and 1
Author - Mishita Kesarwani
"""

def min_max_scaler(value, maxValue, minValue):
  return (value - minValue) / (maxValue - minValue)

""" z_normal function is defined below where we take 2 inputs : dataframe and the column which needs to be normalized
Normalized values after scaling will be mapped from -1 to 1
Author - Krutika Hampannavar"""

def z_normal(df, column):
    return df[column].apply(lambda value : (value - df[column].mean()) / df[column].std())

"""Data normalization function is defined below where we take 4 inputs : entire dataset, 
list of columns that we need to normalize using min-max scaler method,
list of columns that we need to encode as 0 and 1 and appending new columns to the dataset,
list of columns that we need to normalize using z_normal method
Author - Krutika Hampannavar"""


def data_normalization(df, cols_to_normalize=[], cols_to_encode=[], cols_to_znorm=[]):
  for col in cols_to_normalize:
    df[col] = df[col].apply(lambda row: min_max_scaler(row, df[col].max(), df[col].min()))
    for col in cols_to_encode:
      for value in df[col].unique():
        df[col+str(value)] = df[col].apply(lambda row : 1 if row == value else 0)
  for col in cols_to_znorm:
    df[col] = z_normal(df, col)
  return df


"""importing numpy and pandas library and matplotlib"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as mlp


"""Customized Logistic_Regression_Model class is defined below where we have following 4 important functions:
_init__
model_training
update_weight
model_prediction"""

class Logistic_Regression_Model() :

#__init__ function defined below is same as constructor where self is like object of class Logistic_Regression_Model and then,
#value of passed arguments are assigned to the attributes.
#Author - Krutika Hampannavar

 

	def __init__( self, learning_rate, iterations ) :		
		self.learning_rate = learning_rate		
		self.iterations = iterations
		
#model_training function is defined below where we take X_train and Y_train (defined after splitting data) as inputs
#Author - Mishita Kesarwani
 
	def model_training( self, X, Y ) :		
		"""Number of labels=m, Number of feautures=n	"""
		self.m, self.n = X.shape		
		"""Initialising Vector W with zeroes, where n = number of features
        Where W=weight and b=bias"""
		self.W = np.zeros( self.n )		
		self.b = 0		
		self.X = X		
		self.Y = Y
		
		"""Gradient descent - Updating weights for the number of times of iterations"""
				
		for i in range( self.iterations ) :			
			self.update_weight()			
		return self
	
	"""Authors - Mishita Kesarwani and Krutika Hampannavar"""
	def update_weight( self ) :		
		sigmoid_Function = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) )
		
		""" calculating gradient value - Here we are going to start from initail weight, calculate derivative and then update our parameters for total number of iterations until we reach local minima."""	
		diff = ( sigmoid_Function - self.Y.T )		
		diff = np.reshape( diff, self.m )		
		"""In above piece of code we have calculated the difference between predicted and actual values of Y and have stored it in diff variable, and reshaping the vector with number of variables """
		dW = np.dot( self.X.T, diff ) / self.m		
		db = np.sum( diff ) / self.m
        
        #Calculating weight derivative and bias dervative above.
        #Formula for caluculation of weight derivative (dW)=(1/Number of Labels)*(dot product of(transpose of X).(difference of predicted Y and actual Y ))
        #Forula for calculation of bias (db)=(1/Number of Labels)*(sum(difference of predicted Y and actual Y))
        #update weight and bias
		self.W = self.W - self.learning_rate * dW	
		self.b = self.b - self.learning_rate * db
		
		return self
	
	""" model_prediction function is defined below where we are calculating the predicted sigmoid function and checking if the propbability is greater than 0.5 or not
	If probabilty ia greater than 0.5 then 1 is returned , if it is less than 0.5 then we return 0
    Author - Krutika Hampannavar"""
	def model_prediction( self, X ) :	
		Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )		
		Y = np.where( Z > 0.5, 1, 0 )		
		return Y
    #Above function will take features of test data as input and return predicted label value

# main method where we we are splitting data,training our model and making predictions for test data

def main() :
  """Reading input CSV file"""
  path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "wildfires.txt")) 
  datasource = pd.read_csv(path, sep = "\t")
  
  """Shuffling the input data,training model and predicting label from test data 10 times
  Encoding features month and year as 0 or 1. And columns with unique values to data set
  Normalizing all the columns using min max scalar method
  Dropping day column after normalization is completed
  As we have encoded values for columns year and month we will drop the actual columns
  Author - Mishita Kesarwani """
  list_accuracy = []
  for _ in range(10):
    dataset = datasource
    dataset = dataset.sample(frac = 1)
    cols_to_normalize = ["temp", "humidity"]
    cols_to_encode = ["month","year"]
    cols_to_znorm =["rainfall", "drought_code", "buildup_index", "wind_speed"]

    dataset = data_normalization(dataset, cols_to_normalize,cols_to_encode,cols_to_znorm)
    dataset["fire"] = dataset["fire"].apply(lambda row: 1 if row.strip().lower() == "yes" else 0)
    dataset.drop(["year", "day", "month"], axis=1, inplace=True)

    label = dataset["fire"]
    dataset = dataset.drop("fire",axis=1)

    """Converting dataset and label to numpy array"""
    dataset = dataset.to_numpy()
    label = label.to_numpy()
    """Splitting data into train and test set. Where 2/3 of data is training data and 1/3 is testing data"""
    test_size = 0.33
    split_size = -int(test_size * len(dataset))
    X_train, X_test = dataset[:split_size ], dataset[split_size:]
    Y_train, Y_test = label[:split_size ], label[split_size:]
    
    """Initialising logistic regression model by passing learning rate and number of iterations as arguements"""
    model = Logistic_Regression_Model( learning_rate = 0.05, iterations = 10000 )
    
    """Training our model with features and label of training set"""
    model.model_training( X_train, Y_train )
    """Prediction on test set"""
    Y_pred = model.model_prediction( X_test )	
    """Calculating accuracy of the model"""
    count = 0
    exact_classification = 0
   
    """ Accuracy Calculations
    Author- Mishita Kesarwani"""
    
    for count in range( np.size( Y_pred ) ) :
      if Y_test[count] == Y_pred[count] :
        exact_classification = exact_classification + 1
    list_accuracy.append((exact_classification / count ) * 100 )
    print( "Accuracy by our logistic regression model     :", 
          (exact_classification / count ) * 100, "\n" )
  
  avr_Acc = 0.00
  counter = 0
  for i in list_accuracy:
    counter += 1
    avr_Acc = avr_Acc + i 
  print("Overall accuracy:", avr_Acc/10)   

  #Authors - Krutika Hampannavar and Mishita Kesarwani""" 
  mlp.plot(range(10), list_accuracy, label = 'Our model logistic regression')
  #plotting of accuracy of all 10 iterations
  mlp.show()


if __name__ == "__main__" :	
  main()
