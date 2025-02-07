# module-12
Introduction to Classification


## Overview:
- Definition:  Predicts the value of a target variable by building a model based on one or more predictors
- Goal:  To compute the category of data
- Real-World Applications:  Email spam protection, customer churn, conversion prediction
- Algorithm	K-nearest neighbors, decision trees

## Classification vs. Regression
Classification and regression are fundamental types of supervised learning algorithms used in machine learning for predictive modeling based on input data. As you read on the previous page, classification involves predicting categorical labels, aiming to assign inputs to predefined classes or categories. For instance, an email spam detector categorizes emails as either "spam" or "not spam." Regression, on the other hand, predicts continuous numerical values, for example, predicting the price of a house based on features such as size, location, and number of rooms.

Since classification outputs categorical values and regression outputs continuous values, their evaluation metrics differ accordingly. For regression models, evaluation metrics such as mean squared error (MSE) and mean absolute error are commonly used. In contrast, classification models are evaluated using metrics such as accuracy, precision, recall, F1 score, and Receiver Operating Characteristics Area Under the Curve (ROC AUC).

This distinction ensures the appropriate assessment and optimization of models tailored to their specific output types and applications in machine learning.

## Types of Classifier Training

### k-nearest neighbors (KNN)
The k-nearest neighbors (KNN) method is a supervised machine learning algorithm that is simple and easy to implement for solving classification and regression problems. KNN relies on labeled input data to learn a function that produces an appropriate output when given new unlabeled data. As you can observe in the following plot, KNN has produced a tightly clustered area that identifies similarities in the two datasets.

To accomplish classification tasks, the KNN algorithm finds the distance between a given data point and k numbers of other points in the dataset that are close to the initial point. The algorithm then votes for the most prevalent category for each individual point.

One of the most critical steps in KNN is ensuring the model's accuracy. This is typically accomplished by selecting the optimal value for k. There is no best practice for choosing this value, but as a general rule, you should choose a number that best fits your model. Selecting a lower value may result in overfitting, while choosing a higher value may require high computational complexity.

#### Calculating KNN
Definition: KNN is a supervised learning algorithm that predicts the class (for classification) or value (for regression) of a new data point based on the similarity to its k nearest neighbors in the training dataset.

Proximity-Based Prediction:
The objective is to assign the data point to a class (classification) or predict its value (regression) by considering its proximity to existing data points.

KNN does not assume any specific underlying data distribution.

Phases:
- Training Phase: Data preprocessing occurs. Features and corresponding class labels are organized (in the case of classification).
- Prediction Phase: Given an unlabeled data point (testing data), you can:
  - Calculate the distance (e.g., Euclidean distance) between the testing data point and all training data points.
  - Select the k-nearest neighbors (based on the smallest distances).
  - Assign the majority class among the k neighbors to the testing data point.

  (k) is whats called a "complexity control hyperparameter", similar to alpha in linear regression, where,
  - Complexity is inversely related to k
  - Maximum complexity: k = 1 (this can lead to overfitting)
  - Minimum complexity: k = n (which may lead to underfitting)

  So, for k = n, the model prediction is very simple, returning only the most common class in the training set
  and, for k = 1, the model prediction is more complex, returning the label of the closest training point.

  To deturmin the optimal k one can use cross validation comparing the variance with the training error.  When doing so you would use -k for the X-axis. 

  Error is defined as the misclassification rate for the y-axis or the fraction of perdictions that are incorrect.  In our model this would be the misclassifications of (y-hat) which always equals 1 - accuracy.  So for example of the a model is 80% accurate the misclassification rate would be 0.20 or 20%. 
   
##### Using KNeighborsClassifier
KNeighborsClassifier below.

weights : {'uniform', 'distance'} or callable, default='uniform'
    Weight function used in prediction.  Possible values:

    - 'uniform' : uniform weights.  All points in each neighborhood
      are weighted equally.
    - 'distance' : weight points by the inverse of their distance.
      in this case, closer neighbors of a query point will have a
      greater influence than neighbors which are further away.
    - [callable] : a user-defined function which accepts an
      array of distances, and returns an array of the same shape
      containing the weights.

===========================

p : int, default=2
    Power parameter for the Minkowski metric. When p = 1, this is
    equivalent to using manhattan_distance (l1), and euclidean_distance
    (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    See an example: [codio_assignment12_1.ipynb](module-12/edit/main/codio/codio_assignment12_1.ipynb)
