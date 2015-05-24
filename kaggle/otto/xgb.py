import xgboost as xgb
from sklearn.grid_search import GridSearchCV
import os
from sklearn.externals import joblib

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
        

mobj = method()
train_file = "../data/train.csv"
test_file = "../data/test.csv"
mobj.prepare_data(train_file, test_file)

xgb_model = xgb.XGBClassifier(n_estimators=1000)
#clf = GridSearchCV(xgb_model,{'n_estimators': [50,100,200]}, verbose=1)
xgb_model.fit(mobj.X, mobj.y)
if os.path.isdir('../model/xgb_1000') == False: 
    os.mkdir('../model/xgb_1000') 
joblib.dump(xgb_model, '../model/xgb_1000/xgb.pkl')

#print(clf.best_score_)
#print(clf.best_params_)
predictions = xgb_model.predict_proba(mobj.x_test)

f = open("../data/output_xgb_1000.csv", "w")
f.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
i=1
for row in predictions:
    row = map(str, row)
    f.write(str(i)+","+str(",".join(row))+"\n")
    i += 1
f.close()