import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mlxtend.sklearn import SBS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

scr = StandardScaler()
X_std = scr.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=4)

# selecting features
sbs = SBS(knn, k_features=1, scoring='accuracy', cv=5)
sbs.fit(X_std, y)

print('Indices of selected features:', sbs.indices_)
print('CV score of selected subset:', sbs.k_score_)
print('New feature subset:')
print sbs.transform(X)[0:5]

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.show()