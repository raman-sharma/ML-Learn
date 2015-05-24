import numpy as np
import os
import datetime
import pandas as pd
import traceback

from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn.svm import SVC,LinearSVC

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
               
mobj = method()
train_file = "../data/train.csv"
test_file = "../data/test.csv"
mobj.prepare_data(train_file, test_file)

clf = SVC(kernel='rbf',probability=True)
#mobj.evaluate_classifier([clf], ['svc-rbf'])
mobj.predict_test(clf, 'svc_rbf.csv')
