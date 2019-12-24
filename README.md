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

**Categorical Transformations:**
We examined three approaches to handle the fact that we have several distinct values per categorical feature. One-hot encoding all features alone would have led to potentially over a million columns and would not have allowed for a scalable implementation. Based on Brieman's theorem, we first computed the weighted average of the target variable (our click-through rate) for every value within each category. Then we decided to explore three options:
* Split the categorical feature by two bins (high and low) based on the average value of the target variable. A high bin has a value of 0.2 or greater and a low bin has everything else including null values. After creating our bins, we one-hot encoded so that all of our variables would be in the format of 0's and 1's
* Split the categorical feature by four bins (high, middle, low and null) based on the average value of the target variable. A high bin has a value of 0.6 or greater, a medium bin has a value of 0.1 - 0.6, and a low bin has a value below 0.1. We put all null values in their own, missing bin. We also one-hot encoded these bins as well
* Transformed the categorical features to numeric features using the weighted average values of the target variable. We imputed the mean for all missing values

**Integer Transformations:**
We examined two different imputing methods as many of our numeric variables had a high percentage of null values. Therefore, we used two different imputing techniques:
* Replace missing values with the mean of each numeric column
* Replace missing values with zero


## Algorithms Chosen
We chose logistic regression and decision tree as our main algorithms to predict CTR of the Criteo dataset (0 = "no click" = "failure", 1 = "click" = "success"). 

**Logistic Regression:** 
Logistic regression is a method to calculate the probability of a binary result (where the values are 1s or 0s) given some initial values. We calculate logistic regression in terms of "odds": the probability that a particular outcome is a success divided by the probability that it is a failure. We use stochastic gradient descent, where we start at some initial coefficients and calculate initial predictions, then use the errors in those predictions and a learning rate to calculate new coefficients, and repeat until the errors are small enough (according to a pre-defined convergence criteria) or we reach a pre-defined maximum number of iterations. 

With the logistic regression algorithm we tried hypertuning of the elasticNetParam and the regParam. The elasticNetParam mixing parameter is the regularization that takes values in the range of `0, 1`. A parameter of 0 is the L2, ridge regularization penalty, and a parameter of 1 is the L1, lasso regularization penalty. Our model winded up favoring ridge regularization, which indicates that most of our variables impact the model results and have coefficient weights of roughly equal size. The regParam or learning rate is a regularization parameter that takes a value of greater than 0. We saw that in out models a small learning rate making small updates to the model coefficients throughout each iteration performed best

**Decision Tree:** 
Decision trees are one of the most popular machine learning algorithms, mostly for classification. CART (classification and regression tree) divides the data in homogenous subsets using binary recursive partitions. The most discriminative variable is first selected as the root node to partition the data set into branch nodes. The partitioning is repeated until the nodes are homogenous enough to be the final nodes which are called leaves. Two popular feature selection measures that split the data are the information gain (based on entropy) and gini index.

With the decision tree algorithm we tried hypertuning of the number of bins, depth of the tree, and impurity. The maximum number of bins represents the number of node splits at each layer. The maximum depth represents the maximum depth of each decision tree, or the maximum number of levels of each of those trees. Limiting the depth of the trees helps reduce the number of important features. We have to evaluate a balance between the number of nodes within each layer of the tree and maximum layers of the tree. The impurity criterion in this instance refers to the quality of each node split within our tree. The gini criterion was chosen as the best parameter over the entropy criterion. This measures how often a randomly chosen observation within our training data would be mislabeled.


## Model Implementation and Results
**Baseline Model:**
We started out with a baseline model. The goal of our baseline model is to develop a "baseline" threshold for our f1-score; essentially this model represents the f1-score that we would like to beat. We created three different baseline models to cover all potential approaches. Our first baseline model predicts all rows as the majority class of not clicking on the add. The F1 score is 0.86, but we know this is likely to be true with our high amount of class unbalance. Our second baseline model is more balanced and randomly assigns 75% of the predicted values to the majority class of 0, and the other 25% or so of the values to the minority class of 1. The F1 score in this model is 0.73, which is more realistic. Our last baseline model is randomly assigns a 1 or 0 as the prediction. The F1 score in this case is 0.58, which is lower than what we would expect.

**Actual Models:**
Overall, we feel that the models we evaluated appeared to have reasonable F1 scores for initial attempts (at least better than coin flips). For the categorical variables, the weighted value transformations proved to have the best F1 scores on larger datasets, while the high/low binned one-hot encoded transformations led to better performance on smaller datasets. The weighted value transformations had more of an overfitting issue with less data.

For the numeric variables, we observed very little difference between imputing the mean of the column or zeros for all variables. This is likely due to the fact that we are still replacing all null values with one number and that specific number matters less. As a next step, we would leverage the simplicity and performance efficiency of imputing zero, and consider other, more complex imputation methods for further optimizations.

Logistic regression performed better than decision trees based on F1 scores. Generally, decisions are expected to perform better. However, when classes are not well separated, such as in continuous variables, logistic regression models can generalize better. Perhaps the transformation of categories to weight values may be responsible for this observation. However, we will feel that we have not explored this dataset fully to make this a recommendation and would plan to further explore as a next step.

Within the logistic regression models, parameter hypertuning did not improve F1 scores, and in a couple cases actually reduced them. We hypothesize that this is based on the random variability of the cross-validation folds, given that we only computed three folds per model, but may have also been due to poorly-documented features of the Apache Spark libraries that otherwise decreased performance in our case.

**Log Loss:**
In addition to thinking about the F1 scores for all of our models, we also evaluated the log loss of our models. Log Loss is well used in logistics regression since the prediction function is non-linear as it is a sigmoidal function. While not as frequent some analyst have used log loss in decision trees, which we looked at as well. In both instances the prediction is between 0 and 1 where the model uses probability to make this classification. The average of a single observation log loss's is the log loss of the model and should be minimized.

Pragmatically log loss is a useful tool to identify the best model during tuning. Log loss is convex guranteeing a global minima that will represent the best model. We chose to use the actual data to do a graph but a graph could be imputed with the range between 0 and 1 to show all the hypothetical loss for each prediction (1 or 0). This global minima will intersect.

The decision tree log loss actual results tend to be very close to either one or zero and nothing in between. This is akin to Naive Bayes that may be over confident where a tool such as Laplace smoothing would help generalize the model and make log loss more useful. We have not been able to perform Laplace smoothing on our models, but would look into it further in the future.
