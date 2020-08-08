import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

print("\n part a) ")
#data=open('data_alda','r')
dataset_train=pd.read_csv('hw2q1_train.csv')
dataset_test=pd.read_csv('hw2q1_test.csv')

train_labels=dataset_train['Class']
test_labels=dataset_test['Class']

count_R_train=0
count_M_train=0

count_M_test=0
count_R_test=0
for i in dataset_train.values:
    if(i[60]=='R'):
        count_R_train=count_R_train+1
    if (i[60] == 'M'):
        count_M_train = count_M_train + 1


for i in dataset_test.values:
    if(i[60]=='R'):
        count_R_test=count_R_test+1
    if (i[60] == 'M'):
        count_M_test = count_M_test + 1

print("Size of training set is : ", len(dataset_train))

print("Size of test set is : ", len(dataset_test))



print("The no of Rocks in training set is :", count_R_train)
print("The no of Rocks in test set is :", count_R_test)
print("The no of Minerals in training set is :", count_M_train)
print("The no of Minearals in test set is :", count_M_test)

## part b)
print("\n part b) (Normalized data)")
rem_last_col_train=dataset_train.drop('Class', axis=1)
rem_last_col_test=dataset_test.drop('Class', axis=1)



normalized_train = (rem_last_col_train - rem_last_col_train.min()) / (rem_last_col_train.max() - rem_last_col_train.min())
normalized_test = (rem_last_col_test - rem_last_col_train.min()) / (rem_last_col_train.max() - rem_last_col_train.min())



print("\n part b) i (Normalized data)")
cov_train=normalized_train.cov()
print("Coviariance matrix : ")
print(cov_train)
print("Calculated Covariance matrix")
print("Further details to see the coariance matrix, please use the code file attached.")

## part b) ii
print("\n part b) ii (Normalized data)")

print("Shape of covariance matrix of NEW - normalized training data: ",cov_train.shape)
print("Size of covariance matrix of NEW - normalized training data",len(cov_train) )

#eigen_train=np.linalg.eigvals(cov_train.apply(pd.to_numeric, errors='coerce').fillna(0))

#print(np.round(eigen_train,3))


eig_values, eig_vectors = np.linalg.eig(cov_train)
print("Eigen values :")
print(eig_values)
print("Eigen vectors ")
print(eig_vectors)


print("The top 5 eigen values are :")

print(np.round(eig_values[0:5],3))



print("\n part b) iii (Normalized data)")
print("Graph for Eigen Values -- Normalized data")
plt.plot(eig_values)
plt.title("Eigen Values")
plt.ylabel("Eigen Values")
plt.xlabel("Principal Components")
plt.show()


## part b) iv
## applying pcc and knn
p=[2, 4, 8, 10, 20, 40, 60]

## Make a 156 * p matrix where p is no of principal vectors
a=normalized_train
a_test=normalized_test
print("Normalized training ")
print(a)
accu=[]

for i in (2,4,8,10,20,40,60):
    b=eig_vectors[0:i,]
    bt=np.matrix(b)
    btt=bt.T
    print("btt for i", i)
    print(btt)
    sol=np.matmul(a,btt)
    sol_test=np.matmul(a_test,btt)
    print(i)
    print("The sol :")
    print(sol)
    
    var_train = 'train{}'.format(i)
    locals()[var_train] = sol
    
    var_test= 'test{}'.format(i)
    locals()[var_test]=sol_test
    


classifier_2 = KNeighborsClassifier(n_neighbors=3,metric='euclidean')
classifier_2.fit(train2,train_labels)
output_2=classifier_2.predict(test2)
accu.append(accuracy_score(test_labels,output_2))


classifier_4 = KNeighborsClassifier(n_neighbors=3,metric='euclidean')
classifier_4.fit(train4,train_labels)
output_4=classifier_4.predict(test4)
accu.append(accuracy_score(test_labels,output_4))


classifier_8 = KNeighborsClassifier(n_neighbors=3,metric='euclidean')
classifier_8.fit(train8,train_labels)
output_8=classifier_8.predict(test8)
accu.append(accuracy_score(test_labels,output_8))


classifier_10 = KNeighborsClassifier(n_neighbors=3,metric='euclidean')
classifier_10.fit(train10,train_labels)
output_10=classifier_10.predict(test10)
accu.append(accuracy_score(test_labels,output_10))
acc_p10=accuracy_score(test_labels,output_10)


classifier_20 = KNeighborsClassifier(n_neighbors=3,metric='euclidean')
classifier_20.fit(train20,train_labels)
output_20=classifier_20.predict(test20)
accu.append(accuracy_score(test_labels,output_20))


classifier_40 = KNeighborsClassifier(n_neighbors=3,metric='euclidean')
classifier_40.fit(train40,train_labels)
output_40=classifier_40.predict(test40)
accu.append(accuracy_score(test_labels,output_40))

classifier_60 = KNeighborsClassifier(n_neighbors=3,metric='euclidean')
classifier_60.fit(train60,train_labels)
output_60=classifier_60.predict(test60)
accu.append(accuracy_score(test_labels,output_60))






print("\n part 1 in b)iv ---> p=10  (Normalized data)")

df=pd.DataFrame(columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','ActualOutput','PredictedOutput'])

for i in range(0,52):
    df.at[i,'PC1']=test10[i,0]
    df.at[i,'PC2']=test10[i,1]
    df.at[i,'PC3']=test10[i,2]
    df.at[i,'PC4']=test10[i,3]
    df.at[i,'PC5']=test10[i,4]
    df.at[i,'PC6']=test10[i,5]
    df.at[i,'PC7']=test10[i,6]
    df.at[i,'PC8']=test10[i,7]
    df.at[i,'PC9']=test10[i,8]
    df.at[i,'PC10']=test10[i,9]
    df.at[i,'ActualOutput']=test_labels[i]
    df.at[i,'PredictedOutput']=output_10[i]


print("The csv file of 12 columns of Normalized dataset is saved in question1b_p10_norm.csv file")
df.to_csv('question1b_p10_norm.csv',sep=" ")
print("The accuracy when p=10for Normalized data is ", acc_p10)

## part 2 in b) iv - all p's
print("\n part 2 in b) iv all p's (Normalized dataset)")
plt.plot(p,accu)
plt.title("Accuracy v/s Number of PC's")
plt.xlabel("Number of PC's used")
plt.ylabel("Accuracy")
plt.show()

print("\n part 3 in b)iv  (Normalized data)")
print("Looking at the data, no of p's is ideally ____ as the accuracy is low for that")
###########
print("\n-------------For Standardized Data-------------------------------")
## part c) - repeating with standardized data

################


#standardized_train = (rem_last_col_train - rem_last_col_train.min()) / (rem_last_col_train.max() - rem_last_col_train.min())
#standardized_test = (rem_last_col_test - rem_last_col_train.min()) / (rem_last_col_train.max() - rem_last_col_train.min())
from sklearn import preprocessing


standardized_train=pd.DataFrame(preprocessing.scale(rem_last_col_train))
standardized_test=pd.DataFrame(preprocessing.scale(rem_last_col_test))


print("\n part b) i (Standardized data)")
cov_train_stand = standardized_train.cov()
print("Calculated covariance matrix")
print("USe code file to see the covariance matrix")

print("\n part b) ii (Standardized data)")
print("Shape of covariance matrix of NEW - standardized training data: ", cov_train_stand.shape)
print("Size of covariance matrix of NEW - standardized training data", len(cov_train_stand))

# eigen_train=np.linalg.eigvals(cov_train.apply(pd.to_numeric, errors='coerce').fillna(0))

# print(np.round(eigen_train,3))


eig_values_stand, eig_vectors_stand = np.linalg.eig(cov_train_stand)

print("The top 5 eigen values are :")

print(np.round(eig_values_stand[0:5], 3))

print("part b) iii (Standardized data)")
print("Graph for eigen values- Standardized data")
plt.plot(eig_values_stand)
plt.title("Eigen Values")
plt.ylabel("Eigen Values")
plt.xlabel("Principal Components")
plt.show()

print("\n part b) iv (Standardized data)")
## applying pcc and knn

## Make a 156 * p matrix where p is no of principal vectors
a_stand = standardized_train
a_test_stand = standardized_test

accu_stand = []

for i in (2, 4, 8, 10, 20, 40, 60):
    b_stand = eig_vectors_stand[0:i, ]
    bt_stand = np.matrix(b_stand)
    btt_stand = bt_stand.T
    sol_stand = np.matmul(a_stand, btt_stand)
    sol_test_stand = np.matmul(a_test_stand, btt_stand)

    var_train_stand = 'trainstand{}'.format(i)
    locals()[var_train_stand] = sol_stand

    var_test_stand = 'teststand{}'.format(i)
    locals()[var_test_stand] = sol_test_stand

classifier_2_stand = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
classifier_2_stand.fit(trainstand2, train_labels)
output_2_stand = classifier_2_stand.predict(teststand2)
accu_stand.append(accuracy_score(test_labels, output_2_stand))

classifier_4_stand = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
classifier_4_stand.fit(trainstand4, train_labels)
output_4_stand = classifier_4_stand.predict(teststand4)
accu_stand.append(accuracy_score(test_labels, output_4_stand))

classifier_8_stand = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
classifier_8_stand.fit(trainstand8, train_labels)
output_8_stand = classifier_8_stand.predict(teststand8)
accu_stand.append(accuracy_score(test_labels, output_8_stand))

classifier_10_stand = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
classifier_10_stand.fit(trainstand10, train_labels)
output_10_stand = classifier_10_stand.predict(teststand10)
accu_stand.append(accuracy_score(test_labels, output_10_stand))
accu_p10_stand=accuracy_score(test_labels, output_10_stand)
classifier_20_stand = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
classifier_20_stand.fit(trainstand20, train_labels)
output_20_stand = classifier_20_stand.predict(teststand20)
accu_stand.append(accuracy_score(test_labels, output_20_stand))

classifier_40_stand = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
classifier_40_stand.fit(trainstand40, train_labels)
output_40_stand = classifier_40_stand.predict(teststand40)
accu_stand.append(accuracy_score(test_labels, output_40_stand))

classifier_60_stand = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
classifier_60_stand.fit(trainstand60, train_labels)
output_60_stand = classifier_60_stand.predict(teststand60)
accu_stand.append(accuracy_score(test_labels, output_60_stand))



print("\n part 1 in b)iv ---> p=10 (Standardized data)")

df_stand = pd.DataFrame(
    columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'ActualOutput', 'PredictedOutput'])

for i in range(0, 52):
    df_stand.at[i, 'PC1'] = teststand10[i, 0]
    df_stand.at[i, 'PC2'] = teststand10[i, 1]
    df_stand.at[i, 'PC3'] = teststand10[i, 2]
    df_stand.at[i, 'PC4'] = teststand10[i, 3]
    df_stand.at[i, 'PC5'] = teststand10[i, 4]
    df_stand.at[i, 'PC6'] = teststand10[i, 5]
    df_stand.at[i, 'PC7'] = teststand10[i, 6]
    df_stand.at[i, 'PC8'] = teststand10[i, 7]
    df_stand.at[i, 'PC9'] = teststand10[i, 8]
    df_stand.at[i, 'PC10']= teststand10[i, 9]
    df_stand.at[i, 'ActualOutput'] = test_labels[i]
    df_stand.at[i, 'PredictedOutput'] = output_10_stand[i]


print("The csv file of 12 columns for Standardized data is saved in question1b_p10_stand.csv file")

df_stand.to_csv('question1b_p10_stand.csv',sep=" ")
print("The accuracy when p=10 for Standardized datset is ", accu_p10_stand)


print("\n part 2 in b) iv - all p's (Standardized data)")
plt.plot(p, accu_stand)
plt.title("Accuracy v/s Number of PC's")
plt.xlabel("Number of PC's used")
plt.ylabel("Accuracy")
plt.show()

print("\n part 3 in b)iv  (Standardized data)")
print("Looking at the results, I would choose p as  __ as the accuracy for it is high")

print("----------------Standardized over - all operations done ---------------------------------------")
## part d (Comparisions b/w stand and norm datasets)
print("Accuracy of Normalized data ",np.round(accu,2))
print("Average : ", np.mean(accu))
print("Accuracy of Standardized data ",np.round(accu_stand,2))
print("Average :",np.mean(accu_stand))