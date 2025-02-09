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

## Decision Boundaries
### Overview:
To train a classifier on a dataset, you must define a set of hyperplanes. These hyperplanes are called decision boundaries, and they separate the data points into specific classes where the algorithm switches from one class to the next. A data point is more likely to be classified as class A on one side of a boundary and class B on the other.

In the logistic regression example below, a decision boundary is a straight line that separates class A from class B. However, it is difficult in linear models to determine the exact boundary line separating the two classes, so points from class A have also come into the region of class B.

Visualizing decision boundaries in this manner helps demonstrate how sensitive models are to the specific dataset, which can help understand how particular algorithms work and what their limitations are.

### Predict_proba and Decision Thresholds
Once you have selected a value for k, you then have choices to make about the "decision threshold" for your model.

Using Scikit-learn "predict_proba(ten_random_rows[["Income", "Debt"]])" you get 
- the predictions about a class of a given sample
- information about the level of confidence in the model
where the order of the confidence values are in alphabetical order prediction class

### Evaluating classifiers:
Classification performance is measured either by a numeric metric, such as accuracy, or a graphical representation, such as a receiver operating characteristic (ROC) curve. Classification metrics are based on the true positives (TPs), false positives (FPs), false negatives (FNs), and true negatives (TNs) contained in the confusion matrix.

![Alt text](https://github.com/codemonkey1101/module-12/blob/main/images/confusion-matrix.png)

The performance of a model can be evaluated using a variety of metrics. It is critical that you understand what each metric calculates to choose the best evaluation metric for your model. For example, models may be hailed as highly accurate, but depending on the question the model is trying to address, another metric may be more appropriate. The metrics that are typically used to determine the performance of a model are:
- accuracy
- precision
- recall
- F1.

#### Accuracy
Accuracy is the most intuitive measure of performance, as it is simply the ratio of correctly predicted observations to total observations. Accuracy can be deceiving in that it may signal a highly accurate model, but in all actuality, it has some weaknesses. Accuracy is only useful when the dataset is perfectly symmetrical, where values of FNs and FPs are almost identical, with similar costs. Accuracy is useful in cases where you have balanced classes, which implies that equal importance is given to both the positive and negative classes. Accuracy provides an overall view of the model’s performance.

To compute the total accuracy of a model using the values in a confusion matrix
Total accuracy = (TN + TP) / (FN + FP + TP + TN)



#### Precision
Precision is the proportion of accurately predicted positive observations in relation to the total predicted positive observations. High precision is directly correlated to a low FP rate. This metric is use to evaluate the reliability of positive predictions made by a model. It answers the question: “Of all the instances predicted as positive, how many were actually positive?”. It is calculated as:

Precision = TP / (FP + TP​)

Precision helps in avoiding FPs, also known as type I errors. FPs occur when the model predicts positive (e.g., disease) when the actual class is negative (e.g., healthy). In critical scenarios (e.g., medical diagnosis, fraud detection), FPs can have severe consequences. High precision minimizes false alarms and ensures that positive predictions are trustworthy. Precision becomes crucial when classes are imbalanced. It helps prevent the overestimation of positive cases. It shares an inverse relationship with recall. Increasing precision often leads to lower recall (and vice versa). Finding the right balance depends on the specific problem.

Precision matters when diagnosing a rare disease. An FP (predicting disease when the patient is healthy) can cause unnecessary stress and additional tests. Similarly, in the case of email spam detection, high precision ensures that legitimate emails are not incorrectly marked as spam. In the case of credit risk assessment, precision is crucial to avoid approving risky loans.

#### Recall
Recall (a.k.a. sensitivity) is the proportion of correctly predicted positive observations in relation to all of the observations in an actual class. As a result, recall measures the precision with which a model can determine the relevant data.

Recall measures the proportion of actual positive instances (TPs) that the model correctly identifies. It answers the question: “Of all the actual positive cases, how many did the model capture?” It is calculated as:

Recall = TP / (TP + FN)

This metric plays a crucial role in areas such as medical diagnosis and quality control. When identifying diseases, missing a TP can have severe consequences. Recall ensures that the actual positive cases are not overlooked. In quality control, for detecting defects or anomalies, recall helps identify all faulty products, minimizing FNs. Recall emphasizes the ability to find actual positive instances. Recall is crucial when avoiding FNs is a priority.

Balancing precision and recall is crucial in scenarios where both FPs and FNs have significant consequences. Please see the table below for more information on the importance of balancing precision and recall in different scenarios.

This metric plays a crucial role in areas such as medical diagnosis and quality control. When identifying diseases, missing a TP can have severe consequences. Recall ensures that the actual positive cases are not overlooked. In quality control, for detecting defects or anomalies, recall helps identify all faulty products, minimizing FNs. Recall emphasizes the ability to find actual positive instances. Recall is crucial when avoiding FNs is a priority.

Balancing precision and recall is crucial in scenarios where both FPs and FNs have significant consequences.

Example:

Scenario 1: Medical Diagnosis
Precision: High precision ensures reliable positive predictions (minimizing false alarms).
Recall: High recall ensures that actual positive cases are not missed (minimizing FNs).

Scenario 2: Quality Control
Precision: High precision avoids false alarms.
Recall: High recall ensures that all faulty products are identified.

#### Specificity

Specificity = TN / (TN + FP)

#### F1
F1 is the weighted average of both precision and recall. The F1 score is essential because it balances precision and recall, providing a single metric that considers both FPs and FNs. It provides a holistic view of model performance, especially in scenarios where precision and recall need careful consideration.

