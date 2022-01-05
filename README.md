**Logistic Regression:**

Logistic regression is a supervised learning classification algorithm. This method is usually 
used for the datasets which has more than one independent variable that determine an 
outcome. Some of the classification examples are whether movie review is positive or not,
rain or no rain, tumour is Malignant or Benign. There are two types of logistic regression 
binary and multilinear logistic regression. In this assignment we are implementing the binary 
logistic regression on our wildfires data set as we have only two possible outcomes. The 
output of logistic regression is transformed by making use of sigmoid function which returns 
a probability value. This wild fire data set contains attributes such as fire, year, temp, 
humidity, rainfall, drought code, buildup index, buildup index, month, wind_speed. Based on 
features such as year, temp, humidity, rainfall, drought_code, buildup_index, buildup_index,
month, wind_speed we are predicting two classes (fire yes, fire no) of fire for test data.

**Logistic Regression algorithm implementation:**

**Step 1: File preparation**
For a given input file “wildfires.txt”. We modified the text file by converting it into a tab 
separated text file which was then converted into “wildfires.csv”. It is then imported using 
pandas.
**Step 2: Data Preprocessing**
For Data Preprocessing we have created a method named data_normalization where we are 
normalizing and encoding columns of our dataset. Features such as ‘temp’, ’humidity’ are
normalized using min_max_scalar method defined in our code. Other features such as
’rainfall’, ’drought code’, ‘buildup_index’, ‘wind_speed’ are normalized using z_normal
method defined in our code. Independent variables ‘year’ and ‘month’ are encoded, new 
columns are appended to the data set with unique values 0 and 1. The values in our label 
column are converted in to 0’s and 1’s. If it’s a “yes” for fire then the value is set 1. If it’s a 
“no” for fire then the value is set 1. Feature ‘day’ is dropped from the dataset as it was 
irrelevant and had less statistical inference on our dataset, also as per our understanding
month and year have more impact on outcome rather than day. After encoding the columns 
‘Month’ and ‘Year’ we have dropped them from dataset.
**Step 3: Splitting of dataset**
The algorithm is iterated over 10 randomly shuffled splits of training and test data. The 
dataset is converted into numpy array. This is then divided into training and testing data. 1/3 
portion of entire dataset is considered for testing and remaining 2/3 for training of model.
**Step 4: Model Traning**
To train our model we have defined customized class - Logistic_Regression_Model , which 
includes following methods:
We model our data using linear function = wx+b , for Logistic regression we don’t want 
continuous values we want probability , for that we use sigmoid function (1/(1+(e^(-x))) , x in 
this case is our linear model this will give probability between 0 and 1
a) __init__: 
This method acts like a constructer of our class where we are initializing objects of 
class by setting the values of learning rate and number of iterations.
b) **update_weight :**
Vector W consists of real numbers that are associated with input feature X. Each 
weight value represents how significant that input feature is for classification. The 
bias is another value that is added to weighted inputs
Here we are initializing vector W with zero n*1 matrix. Initializing X with X_train 
values and Y with Y_train values.
Where:
W is n*1 matrix
X is a n*m matrix
Y is 1* m matrix
n is number of features 
m is number of labels.

In training the dataset we are using sigmoid function and gradient descent function. Initially 
as part of training model we update the weight vector W with all zeros. The sigmoid function
(hypothesis of logistic regression) is used to set data where the independent variable(s) can 
take any real value, and the dependent variable is either 0 or 1. The sigmoid function value 
ranges 0 and 1. Gradient descent starts from initial weight, then derivative is calculated and 
then we update our parameters for total number of iterations until we reach local minima.
Cost function is used for error representation. We cannot use the cost function which is used 
for linear regression as that ends up in non-convex function. But for logistic regression we 
use need convex function. Gradient descent sets value of W in such a way that it always 
converges to local minima. Later weight derivative and bias derivative are calculated.

**Formula used:**
Sigmoid function: 1/(1 + (𝑒^(−(𝑤𝑥+𝑏)))
Gradient Descent : 𝑊 − 𝛼(𝑑𝑊) where α is learning rate.
                   𝑏−∝ (𝑑𝑏)
                   
**Step 5: Model Testing**
We feed the testing data set for prediction function. 0.5 value acts as decision boundary for 
deciding the class for our test set. Suppose if the value is considered as 0.7 then it would 
mean that it is more in support of class 1 rather than 0. The hypothesis function goes as 
follows:
𝑍 >=0.5 → class is 1
Z<0.5 → class is 0
If the probabilistic value of variable ‘Z’ is greater than 0.5 then the class value is set to 1. If 
the value of variable ‘Z’ is less than 0.5 then the class value is set 0. The class 1 is for fire –
‘yes’ and the class 0 is for fire – ‘no’.
**Step 6: Accuracy** 
We calculate the accuracy of the models by considering the counts of existing labels and 
predicted labels. Average accuracy of our model is calculated by considering the sum of all 
accuracy’s and dividing it by number of times the model is run.
Comparison of our model with the sklearn logistic regression:
For both our model and sk learn logistics regression we have made use of same preprocessing 
techniques. Preprocessed train feature and train label are fed into sklearn fit model. Using the 
existing predict function of sklearn logistics regression we predict the test labels. For both the 
models the data is shuffled 10 times

**Results:**
The accuracies of all 10 iterations of our model and sk learn logistics regression model are calculated 
and average accuracies of those iterations for both our model and sk learn model is calculated. It is 
observed the for our model we are getting average accuracy of 84.54% approximately and for the Sk 
learn model it is around 86.96 % approximately. These both models have almost same average 
accuracies.
