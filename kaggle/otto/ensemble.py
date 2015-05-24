from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV
from sklearn.linear_model import Perceptron

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid 

from mlxtend.sklearn import EnsembleClassifier

import numpy as np
import datetime
import pandas as pd
import traceback
from sklearn.externals import joblib
import os

class method:
    def __init__(self):
        self.header = []
        self.X = []
        self.y = []
        self.x_test = []
        
    def prepare_data(self, train_file, test_file):
        f = open(train_file, 'r')
        for line in f:
            if "target" in line:
                self.header = line.strip().split(',')
            else:
                row_x = line.strip().replace("Class_","").split(',')
                row_x = [ int(x) for x in row_x ]
                self.X.append(row_x[1:93])
                self.y.append(row_x[94])
        f.close()
        
        f = open(test_file, 'r')
        for line in f:
            if "id" not in line:
                row_x = line.strip().split(',')
                row_x = [ int(x) for x in row_x ]
                self.x_test.append(row_x[1:93])
        f.close()

    def ensemble_classifier_cross_validation(self, clfs_arr, labels, voting_flag='hard'):
        np.random.seed(1919)                   
        eclf = EnsembleClassifier(clfs=clfs_arr, voting=voting_flag)
        
        clfs_arr = [eclf] + clfs_arr
        labels = ['Ensemble'] + labels
        
        for clf, label in zip(clfs_arr[:1], labels[:1]):
            print str(label)+" started on "+str(datetime.datetime.now())
            scores = cross_validation.cross_val_score(clf, self.X, self.y, cv=10, scoring='accuracy')
            print("\tAccuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
            print str(label)+" ended on "+str(datetime.datetime.now())
            print
        
    def evaluate_classifier(self, clfs_arr, labels):
        np.random.seed(1919)          
        for clf, label in zip(clfs_arr, labels):
            try:
                print str(label)+" started on "+str(datetime.datetime.now())
                os.mkdir('../model/'+label)
                clf.fit(self.X, self.y)
                joblib.dump(clf, '../model/'+label+'/'+label+'.pkl')
                scores = cross_validation.cross_val_score(clf, self.X, self.y, cv=2, scoring='accuracy')
                print("\tAccuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
                print str(label)+" ended on "+str(datetime.datetime.now())
            except:
                print traceback.format_exc()
                continue
            print

    def find_tuning_weights(self, clfs_arr):
        np.random.seed(1919)
        df = pd.DataFrame(columns=('w0', 'w1', 'w2', 'w3', 'mean', 'std'))        
        i = 0
        for w0 in range(2,5):
            for w1 in range(2,5):
                for w2 in range(2,5):
                    for w3 in range(2,5): 
                        try:                   
                            if len(set((w0,w1,w2,w3))) == 1: # skip if all weights are equal
                                continue
                            
                            eclf = EnsembleClassifier(clfs=clfs_arr, voting='soft', weights=[w0,w1,w2,w3])
                            scores = cross_validation.cross_val_score(estimator=eclf, X=self.X, 
                                                            y=self.y, cv=10, scoring='accuracy', n_jobs=1)                    
                            df.loc[i] = [w0, w1, w2, w3, scores.mean(), scores.std()]
                            print df.loc[i]
                        except:
                            print traceback.format_exc()
                            continue
                        i += 1
                        
            df.sort(columns=['mean', 'std'], ascending=False)
            print df 
        
    def predict_test(self,clf, tag):
        np.random.seed(1919) 
        if os.path.isdir('../model/'+tag) == False: 
            os.mkdir('../model/'+tag)   
        print "Dir made : "+str(datetime.datetime.now())
        
        print "Fit Started : "+str(datetime.datetime.now())
        clf.fit(self.X, self.y)    
        
        print "Dump Started : "+str(datetime.datetime.now())    
        joblib.dump(clf, '../model/'+tag+'/'+tag+'.pkl')
        
        print "Prediction Started : "+str(datetime.datetime.now())
        output_arr = clf.predict_proba(self.x_test)
        
        f = open("../data/output_"+str(tag), "w")
        f.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
        i=1
        for row in output_arr:
            row = map(str, row)
            f.write(str(i)+","+str(",".join(row))+"\n")
            i += 1
        f.close()
        
        print "ALL DONE : "+str(datetime.datetime.now())
        
################ MAIN ################

mobj = method()
train_file = "../data/train.csv"
test_file = "../data/test.csv"
mobj.prepare_data(train_file, test_file)

#clf1 = LogisticRegression()
clf11 = BaggingClassifier(n_estimators=1000)
clf12 = ExtraTreesClassifier(n_estimators=1000)
clf14 = RandomForestClassifier(n_estimators=1000)

clfs_arr = [clf11,clf12,clf14]
#weights_arr = [1,3,3,3]
#eclf = EnsembleClassifier(clfs=clfs_arr, voting='soft',weights=weights_arr)
eclf = EnsembleClassifier(clfs=clfs_arr, voting='hard')
mobj.predict_test(eclf, 'eclf_1000.csv')


#clf13 = GradientBoostingClassifier()
#clf6 = RidgeClassifierCV()
#clf2 = LogisticRegressionCV()
#clf15 = OneVsOneClassifier()
#clf16 = OneVsRestClassifier()
#clf17 = OutputCodeClassifier()
#clf20 = SVC(kernel='rbf', verbose=3)
#clf21 = SVC(kernel='linear')
#clf22 = SVC(kernel='poly')
#clf23 = SVC(kernel='sigmoid')
#clf24 = SVC(kernel='precomputed')
#clf25 = LinearSVC()
#clf26 = NuSVC()
#clf27 = KNeighborsClassifier()
#clf28 = RadiusNeighborsClassifier()
#clf29 = NearestCentroid()
#clf5 = RidgeClassifier()
#clf7 = GaussianNB()
#clf8 = MultinomialNB()
#clf9 = BernoulliNB()
#clf19 = DecisionTreeClassifier()
#clf3 = SGDClassifier(n_iter=100)
#clf4 = PassiveAggressiveClassifier(n_iter=100)
#clf10 = AdaBoostClassifier(n_estimators=100)
#clf18 = Perceptron(n_iter=100)

