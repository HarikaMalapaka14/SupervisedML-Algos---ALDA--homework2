import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

id=list(range(1,11))
x1=[27,-6,2,36,-8,40,35,30,20,-1]
x2=[6,2,2,2,4,2,4,2,6,4]
Label=['N','N','P','P','N','P','N','P','P','N']

df=pd.DataFrame(list(zip(id,x1,x2,Label)))
df.columns=['ID','x1','x2','Class']

training_feat_1=df.iloc[[0,1,2,3,5,6,7,8,9],[1,2]]
training_labels_1=df.iloc[[0,1,2,3,5,6,7,8,9],[3]]
training_feat_2=df.iloc[[0,1,2,3,4,5,6,7,8],[1,2]]
training_labels_2=df.iloc[[0,1,2,3,4,5,6,7,8],[3]]

testing_feat_1=df.iloc[[4],[1,2]]
testing_feat_2=df.iloc[[9],[1,2]]


from sklearn.neighbors import KNeighborsClassifier
classifier_1 = KNeighborsClassifier(n_neighbors=3,metric='euclidean')
classifier_1.fit(training_feat_1,training_labels_1)

classifier_2 = KNeighborsClassifier(n_neighbors=3,metric='euclidean')
classifier_2.fit(training_feat_2,training_labels_2)

## part a)  5 and 10's nearest neighbors
set_5=(classifier_1.kneighbors(testing_feat_1,3))
set_10=(classifier_2.kneighbors(testing_feat_2,3))
print("The neighbors for 5th point : ")
print(set_5)
print(set_10)


## part a)


print("The nearest 3 neighbors of 5th and 10th point are respectively :")
print(set_5[1], set_10[1])
print("As 0 indexing was done,we changed the point identification numbers in the report.")
## part b) loo cv error

#cls= KNeighborsClassifier(n_neighbors=3,metric='euclidean',)

#cls=KNeighborsClassifier(n_neighbors=3,metric='euclidean')
accu=[]
for i in range(0,10):
    print("LOOCV : ",i+1)
    var_cls='cls{}'.format(i)
    locals()[var_cls]=KNeighborsClassifier(n_neighbors=1,metric='euclidean')
    r=list(range(0,10))
    test=i
    train=[]
    for k in r:
        if(k!=i):
            train.append(k)

   # print(train)
    #print(test)

    train_f=df.iloc[train,[1,2]]
    train_l=df.iloc[train,[3]]
    test_f=df.iloc[test,[1,2]]
    test_l=df.iloc[test,[3]]

    locals()[var_cls].fit(train_f,train_l)
    a=locals()[var_cls].predict(test_f)

    print("Predicted ",a)
    print("Actual",test_l)
    ac=accuracy_score(test_l,a)
    #print(a,test_l)
    accu.append(ac)

incorrects=0
for i in accu:
    if(i==0):
        incorrects=incorrects+1


print("The leave-one-out cross-validation error of 1-NN on this dataset is ", (incorrects/10)*100)

## part c) 5 fold on 3-knn


cv_set_1=[]
cv_set_2=[]
cv_set_3=[]
cv_set_4=[]
cv_set_5=[]



for i in range(0,10):
    j=i%5
    #print(j)
    if(j==0):
        cv_set_1.append(i)
    if(j==1):
        cv_set_2.append(i)
    if(j==2):
        cv_set_3.append(i)
    if(j==3):
        cv_set_4.append(i)
    if(j==4):
        cv_set_5.append(i)




print("The 5 test sets are")

print(cv_set_1)
print(cv_set_2)
print(cv_set_3)
print(cv_set_4)
print(cv_set_5)

errors=0

predicted=[]
actual=[]
for i in range(1,6):
    print("Fold",i)
    var_knn='knn_{}'.format(i)
    locals()[var_knn]=KNeighborsClassifier(n_neighbors=3,metric='euclidean')
    trainset=[]
    testset=[]
    var_sets='cv_set_{}'.format(i)
    testset=locals()[var_sets]
    r_knn3 = list(range(0, 10))
    for k in r_knn3:
        if(k not in testset):
            trainset.append(k)

    #print("training set", trainset)
    #print("testing set ", testset)
    tr_f=df.iloc[trainset,[1,2]]
    tr_lab=df.iloc[trainset,[3]]
    tes_f=df.iloc[testset,[1,2]]
    tes_lab=df.iloc[testset,[3]]
    locals()[var_knn].fit(tr_f,tr_lab)

    var_ot='out_{}'.format(i)
    locals()[var_ot]=locals()[var_knn].predict(tes_f)
    print("Predicted Values",locals()[var_ot])
    print("Actual Values",tes_lab)
    #print("The neighbours are :")
    #print(locals()[var_knn].kneighbors(tes_f,3))


print("As we can see, there are 7 points whose predicted value is not equal to actual value.")
print("Therefore the error is 7/10 = 0.7 or 70%")


