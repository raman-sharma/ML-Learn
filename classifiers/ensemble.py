from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import datasets
from mlxtend.sklearn import EnsembleClassifier

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

np.random.seed(123)

################################
# Initialize classifiers
################################

clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = GaussianNB()

################################
# Initialize EnsembleClassifier
################################

# hard voting    
eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='hard')

# soft voting (uniform weights)
# eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft')

# soft voting with different weights
# eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[1,2,10])



################################
# 5-fold Cross-Validation
################################

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):

    scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))