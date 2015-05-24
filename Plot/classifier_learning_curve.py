from mlxtend.evaluate import plot_learning_curves
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Loading some example data
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=2)

clf = DecisionTreeClassifier(max_depth=1)

plot_learning_curves(X_train, y_train, X_test, y_test, clf, kind='training_size')
plt.show()

plot_learning_curves(X_train, y_train, X_test, y_test, clf, kind='n_features')
plt.show()