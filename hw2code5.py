import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import graphviz
from sklearn import tree




dataset=pd.read_csv('hw2q5.csv')
train=dataset[['patient age','spectacle prescription','astigmatic','tear production rate','Class']].reset_index(drop=True)
test=dataset[['Class']]
clf = DecisionTreeClassifier(criterion='entropy')


cv_set_1=[]
cv_set_2=[]
cv_set_3=[]
cv_set_4=[]
cv_set_5=[]


for i in range(0,24):
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


accur_dt=[]
accur_nb=[]

try:
    for i in range(1,6):
        #print(i)
        var_data = 'cv_set_{}'.format(i)
        a=(locals()[var_data])
        b=list(range(0,24))
        c=[]
        for index in b:
            if(index not in a):
                c.append(index)
        #print(c,a)
        training_feat=train.iloc[c,[0,1,2,3]]
        training_label=train.iloc[c,[4]]

        testing_feat=train.iloc[a,[0,1,2,3]]
        testing_label=train.iloc[a,[4]]
    ## decision tree
        #print(training)
        var_clas='clf{}'.format(i)
        locals()[var_clas]=DecisionTreeClassifier(criterion='entropy')
        hot_train=pd.get_dummies(training_feat)
        print("Data going to DT is ")
        print(hot_train)
        hot_test=pd.get_dummies(testing_feat)
        locals()[var_clas].fit(hot_train,training_label)   ############# fit_transform to fit
        out=locals()[var_clas].predict(hot_test)
        sc=accuracy_score(testing_label,out)
        accur_dt.append(sc)
        print("Model of", i,"th fold Decision tree:")
        print("training set :")
        print(training_feat)
        print("Training labels :")
        print(training_label)
        print("Testing Features :")
        print(testing_feat)
        print("Testing Labels :")
        print(testing_label)
        print("Predictions :")
        print(out)
        print("Accuracy :")
        print(sc)
        print("Confusion Matrix")
        print(confusion_matrix(testing_label,out))
        print("Classification Report")
        print(classification_report(testing_label,out))

        ## Naive Bayes
        # print(training)
        var_naive = 'naive{}'.format(i)
        locals()[var_naive] =  GaussianNB()

        locals()[var_naive].fit(hot_train, training_label)
        out2 = locals()[var_naive].predict(hot_test)
        sc2 = accuracy_score(testing_label, out2)

        accur_nb.append(sc2)
        ###############
        print("Model of", i, "th fold for Naive Bayes:")
        print("training set :")
        print(training_feat)
        print("Training labels :")
        print(training_label)
        print("Testing Features :")
        print(testing_feat)
        print("Testing Labels :")
        print(testing_label)
        print("Predictions :")
        print(out2)
        print("Accuracy :")
        print(sc2)
        post = locals()[var_naive].predict_proba(hot_test)
        print("Posterior Probability/Classification")
        print(np.round(post,4))
        print("Confusion Matrix : ")
        print(confusion_matrix(testing_label, out2))
        print("Classification Report : ")
        print(classification_report(testing_label, out2))


except DeprecationWarning:
    pass
print("The 5 fold CV accuracies for Decision Tree and Naive Bayes are : ")
print((np.round(accur_dt,2)))
print((np.round(accur_nb,2)))
print("DT", np.mean(accur_dt))
print("NB", np.mean(accur_nb))

print("The final model used is training the full dataset using Naive Bayes classifier as it seems to be a better classifier")
final_training_feat=dataset[['patient age','spectacle prescription','astigmatic','tear production rate']].reset_index(drop=True)
final_training_feat_2=pd.get_dummies(final_training_feat)
final_training_lab=dataset[['Class']]
final=GaussianNB()
final.fit(final_training_feat_2,final_training_lab)
out_final =final.predict(final_training_feat_2)
sc_final = accuracy_score(final_training_lab, out_final)


post_final = final.predict_proba(final_training_feat_2)
print("Posterior Probability/Classification of Final Model (Naive Bayes) :")
print(np.round(post_final,4))
print("Confusion Matrix  for Final model (Naive Bayes): ")
print(confusion_matrix(final_training_lab,out_final))
print("Classification Report for Final model(Naive Bayes) : ")
print(classification_report(final_training_lab,out_final))

print("Respresenting DT")
"""
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
features=['a','b','c','d']
target=['Class']

import graphviz
dot_data_1 = tree.export_graphviz(clf1, out_file=None)
graph_1 = graphviz.Source(dot_data_1)
graph_1.render("Decis")
dot_data_1 = tree.export_graphviz(clf1, out_file=None,
                         feature_names=features,
                         class_names=target)


#graph = graphviz.Source(dot_data)
#print(graph.source)
#graph.render('test-output/round-table.gv', view=True)

dot_data_2 = tree.export_graphviz(clf2, out_file=None)
graph_2 = graphviz.Source(dot_data_2)
graph_2.render("DT2")
dot_data_2 = tree.export_graphviz(clf2, out_file=None,
                         feature_names=features,
                         class_names=target,
                         filled=True, rounded=True,
                         special_characters=True)

dot_data_3 = tree.export_graphviz(clf3, out_file=None)
graph_3 = graphviz.Source(dot_data_3)
graph_3.render("DT3")
dot_data_3 = tree.export_graphviz(clf3, out_file=None,
                         feature_names=features,
                         class_names=target,
                         filled=True, rounded=True,
                         special_characters=True)

dot_data_4 = tree.export_graphviz(clf4, out_file=None)
graph_4 = graphviz.Source(dot_data_4)
graph_4.render("DT4")
dot_data_4 = tree.export_graphviz(clf4, out_file=None,
                         feature_names=features,
                         class_names=target,
                         filled=True, rounded=True,
                         special_characters=True)

dot_data_5 = tree.export_graphviz(clf5, out_file=None)
graph_5 = graphviz.Source(dot_data_5)
graph_5.render("DT5")
dot_data_5 = tree.export_graphviz(clf5, out_file=None,
                         feature_names=features,
                         class_names=target,
                         filled=True, rounded=True,
                         special_characters=True)
"""
feat=dataset.columns.values[0:5]
print(feat)
def print_decision_tree_graph(clf,index,df_feature_names):
    dot_data = tree.export_graphviz(clf, out_file=None,feature_names=df_feature_names,class_names=['CLASS'] )
    graph = graphviz.Source(dot_data)
    print("generated")
    graph.render("optic {}".format(index))

print_decision_tree_graph(clf1,1,feat)