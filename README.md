# click-through-rate-prediction
Machine learning project using several million rows of data and computing algorithms in parallel to predict the Criteo platform click-through rate. Based on the former Kaggle competition: http://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use/.

## Project Goal
Click-through rate (CTR) is a measure of how often users click on a specific link among those who view a page, email, or advertisement. The primary goal of our project is to build binary classification models to predict CTR by implementing two popular machine learning algorithms for binary classifications - logistic regression and decision trees. The specific research questions are:
* Which machine learning algorithm produce the best predictions of CTR?
* What metric(s) should we consider when we are defining "best" predictions of CTR?
* How can we implement scalable machine learning algorithms that efficiently handle a large amount of data?

To answer these questions, we analyzed the CTR for the Criteo Advertising Data. The data includes a target variable that indicates if an ad was clicked (1) or not clicked (0), 13 integer features and 26 categorical features. 

We have decided to focus on F1 score as our primary measure of success because it takes precision (ratio of the true positives to true and false positives) and recall (ratio of the true positives to true positives and false negatives) into account. Therefore, the F1 score is essentially taking both false positives and false negatives into account without needing to weight them equally. This type of scoring system to evaluate predictive performance works best when classes are not balanced, which is the case with our data.


## Feature Engineering
**Categorical Variables:**
First, we took the weighted average of the dependent variable (i.e. CTR) for each value within the categorical variables and then imputed these values as the categorical values themselves. For all null values, we imputed the mean of the weighted values in the category. This method essentially turned the categorical variables into numeric variables, which resulted in a bit of overfitting on the toy dataset and a much better performance on the full dataset without further transformation. 

To try and combat any overfitting issues, we both "binned" and "one-hot encoded" the categorical variables. We created feature selections of only two binned categories: a high value (0.2 or greater) and a low value of everything else, which also included null values and then one-hot encoded. We also created a selection of four binned categories: a high value (0.6 or greater), a medium value (0.1 - 0.6), a low value (below 0.1), and null value category and then one-hot encoded. Overall, we had three different transformations of categorical variables that we fed into the model (weighted value and two types of categorical binning).

**Integer Variables:**
We noticed a wide range of values between the integer variables themselves as well as a large right-skew on almost all variables. First, we took the natural log transformation to help normalize our distributions and then normalized all variables on a scale between 0 and 1. Next, we performed two different different imputations for the null values: the mean of the variable and zero.


## Algorithms Chosen
We chose logistic regression and decision tree as our main algorithms to predict CTR of the Criteo dataset (0 = "no click" = "failure", 1 = "click" = "success"). 

Logistic regression is a method to calculate the probability of a binary result (where the values are 1s or 0s) given some initial values. We calculate logistic regression in terms of "odds": the probability that a particular outcome is a success divided by the probability that it is a failure. We use stochastic gradient descent, where we start at some initial coefficients and calculate initial predictions, then use the errors in those predictions and a learning rate to calculate new coefficients, and repeat until the errors are small enough (according to a pre-defined convergence criteria) or we reach a pre-defined maximum number of iterations.

Decision trees are one of the most popular machine learning algorithms, mostly for classification. CART (classification and regression tree) divides the data in homogenous subsets using binary recursive partitions. The most discriminative variable is first selected as the root node to partition the data set into branch nodes. The partitioning is repeated until the nodes are homogenous enough to be the final nodes which are called leaves. Two popular feature selection measures that split the data are the information gain (based on entropy) and gini index.


## Model Implementation
**Baseline Model:**
We started out with a baseline model. The goal of our baseline model is to develop a "baseline" threshold for our f1-score; essentially this model represents the f1-score that we would like to beat. We created three different baseline models to cover all potential approaches. Our first baseline model predicts all rows as the majority class of not clicking on the add. The F1 score is 0.86, but we know this is likely to be true with our high amount of class unbalance. Our second baseline model is more balanced and randomly assigns 75% of the predicted values to the majority class of 0, and the other 25% or so of the values to the minority class of 1. The F1 score in this model is 0.73, which is more realistic. Our last baseline model is randomly assigns a 1 or 0 as the prediction. The F1 score in this case is 0.58, which is lower than what we would expect.
