#!/usr/bin/python

from sklearn.grid_search import GridSearchCV
from sklearn import svm
from numpy import genfromtxt, savetxt
import Util_Poker

def main():
    #clf = svm.SVC(gamma=0.00101, C=99)    
    #clf = svm.SVC(kernel='linear', C=10, gamma=1, degree=1)
    clf = svm.SVC(kernel='poly', C=100, gamma=0.1, degree=1, max_iter=1000000000, verbose=3)
    #clf = svm.SVC(kernel='poly', C=0.01, gamma=1, degree=2)
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('trainPrep.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = [x[-1] for x in dataset]
    train = [x[:-1] for x in dataset]
    test = genfromtxt(open('testPrep.csv','r'), delimiter=',', dtype='f8')[1:]
    test =  [x[1:] for x in test]

    svc=clf.fit(train, target)

    predicted = []
    i = 1
    for x in test:
        y = clf.predict(x)
        predicted.append([i,y])
        i += 1
        
    filename = uniqueNames.generateUniqueFilename("svc_6.csv")
    print "Generating submission: " + filename
    savetxt(filename, predicted, delimiter=',', fmt='%d,%d', 
            header='id,hand', comments = '')


def shuffle_file(old_file, new_file):
    import random

    with open(old_file, 'rb') as infile:
        lines = infile.readlines()
    
    random.shuffle(lines[1:])
    
    with open(new_file, 'wb') as outfile:
        outfile.writelines(lines)
    
        
#################### Grid Search ################
"""
for i in range(100):
    shuffle_file('trainPrep_shuff.csv', 'trainPrep_shuff.csv')
    print "Shuffle "+str(i)

dataset = genfromtxt(open('trainPrep_shuff.csv','r'), delimiter=',', dtype='f8')[1:]    
target = [x[-1] for x in dataset]
train = [x[:-1] for x in dataset]
'''
test = genfromtxt(open('testPrep.csv','r'), delimiter=',', dtype='f8')[1:]
test =  [x[1:] for x in test]
'''
clf = svm.SVC()
params = {'C': [0.01, 0.1, 1, 10, 100, 500, 1000], 'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
          'degree' : [1,2,3,4,5], 'gamma' : [1,0.1,0.001,0.0001, 0.00001, 0.000001]}
grid = GridSearchCV(estimator=clf, param_grid=params, cv=5, scoring="precision")
grid.fit(train[:5000], target[:5000])

for params, mean_score, scores in grid.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
        % (mean_score, scores.std() / 2, params))

"""
main()  
