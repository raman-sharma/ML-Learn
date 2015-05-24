def get_arr(file_name):
    f = open(file_name, 'r')
    X = []
    for line in f:
        if "id" not in line:
            row_x = line.strip().split(',')
            row_x = [ float(x) for x in row_x ]
            X.append(row_x[1:])
    f.close()
    return X

ensemble_prob_1 = get_arr('../data/output_eclf_500.csv') ### 80
gbc_prob_1 = get_arr('../data/output_gbc_500.csv')       ### 80
xgb_prob_1 = get_arr('../data/output_xgb_500.csv')
ensemble_prob = get_arr('../data/output_eclf_1000.csv') ### 80
nn_prob = get_arr('../data/output_NN_500.csv')         ### 80
gbc_prob = get_arr('../data/output_gbc_1000.csv')       ### 80
xgb_prob = get_arr('../data/output_xgb_1000.csv')       ### 80


print len(ensemble_prob)
print len(nn_prob)
print len(gbc_prob)
print len(xgb_prob)

avg_prob = []
for i in range(0,144368):
    tmp_x = []
    for j in range(0,9):
        sum_x = (ensemble_prob_1[i][j] + gbc_prob_1[i][j] + xgb_prob_1[i][j] + ensemble_prob[i][j]*2 + nn_prob[i][j] + gbc_prob[i][j]*2 + xgb_prob[i][j]*2) / 10.0
        tmp_x.append(sum_x)
    avg_prob.append(tmp_x)
    if i%1000 == 0:
        print "Completed "+str(i)

f = open("../data/output_avg_1000_500.csv", "w")
f.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
i=1
for row in avg_prob:
    row = map(str, row)
    f.write(str(i)+","+str(",".join(row))+"\n")
    i += 1
f.close()
