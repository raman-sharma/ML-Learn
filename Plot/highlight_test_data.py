from sklearn.cross_validation import train_test_split
from mlxtend.evaluate import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# Loading some example data
iris = datasets.load_iris()
X = iris.data[:, [0,2]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 

# Training a classifier
svm = SVC(C=0.5, kernel='rbf')
svm.fit(X_train, y_train)

# Plotting decision regions

plot_decision_regions(X, y, clf=svm, X_highlight=X_test, res=0.02, legend=2)

# Adding axes annotations
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.title('SVM on Iris')
plt.show()