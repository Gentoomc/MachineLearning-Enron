#!/usr/bin/python

import sys
sys.path.append("../tools/")
import numpy as np
import pickle

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier,dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm   

######## Loading data & removing outliers
def load_data():
    # Load the dictionary containing the dataset
    data_dict = pickle.load(open("final_project_dataset.pkl", "r"))
    # Remove the large outlier
    data_dict.pop('TOTAL')
    my_dataset = data_dict
    return my_dataset
def transform_features(feature_list, dataset, feature_name):
    import math
    for each_person in dataset.values():
        if (each_person[feature_list[0]] != 'NaN') and (each_person[feature_list[1]] != 'NaN') and (each_person[feature_list[2]] != 'NaN'):
            each_person[feature_name] = (each_person[feature_list[0]] + each_person[feature_list[1]] + each_person[feature_list[2]])*10000
        else:
            each_person[feature_name] = 'NaN'
    return dataset
def dump(clf, my_dataset, features_list):
    dump_classifier_and_data(clf, my_dataset, features_list)
    return

my_dataset = load_data()

new_feature1 = ['other','total_payments','restricted_stock']
new_feature2 = [ 'from_this_person_to_poi','shared_receipt_with_poi','from_messages']

final_feature_set = [
'poi',
'bonus',
'exercised_stock_options',
'expenses',
'feature1',
'feature2'
]      


transformed_dataset = transform_features(new_feature1, my_dataset,feature_name ='feature1')
transformed_dataset = transform_features(new_feature2, my_dataset,feature_name ='feature2')

clf = tree.DecisionTreeClassifier(min_samples_split=2)

test_classifier(clf, my_dataset, final_feature_set)


dump(clf, my_dataset, final_feature_set)




#### Create a new feature and modify data_dict
def pca_features(pca_feature_list, my_dataset):

    data = featureFormat(my_dataset, pca_feature_list,remove_NaN=True, remove_all_zeroes=False, remove_any_zeroes=False, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    new_features = pca.fit_transform(features)

    for index, each_person in enumerate(my_dataset.values()):
        each_person['first_pc'] = new_features[index][0]
        each_person['second_pc'] = new_features[index][1]

    return my_dataset



######## Create different combinations of features
def create_feature_combinations(possible_features_list):
    master_list_features = []
    for i in possible_features_list:
        child_list_features = ['poi', 'exercised_stock_options', 'other', i]
        master_list_features.append(child_list_features)
    return master_list_features

########tunes parameters and returns the lists of best algorithm parameters
def parameter_tuning(my_dataset, each_feature_set):
    from sklearn.cross_validation import StratifiedShuffleSplit
    from sklearn.grid_search import GridSearchCV

    #validation, evaluation for parameter tuning 
    data = featureFormat(my_dataset, each_feature_set, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
    score = 'recall'

    # for score in scores:
        # tune SVC parameters
    svm_tuned_parameters = [{'kernel': ['rbf'], 'C': [1, 3, 10],'degree':[2,3], 'gamma':[10**.1,10**.2]}]
    svm_clf = GridSearchCV(svm.SVC(), svm_tuned_parameters, scoring=score)
    # tune  Decision Tree parameters

    min_samples_split = range(2,100,10)
    min_samples_leaf = range(1,30,10)
    DT_tuned_parameters = [{'min_samples_split': min_samples_split, 'max_features': [None, 'auto'], 'min_samples_leaf': min_samples_leaf}]
    dt_clf = GridSearchCV(tree.DecisionTreeClassifier(), DT_tuned_parameters, scoring=score)
    bestDT = []
    bestSVM = []
    for train_idx, test_idx in cv: 
        features_train,features_test,labels_train,labels_test = [],[],[],[]
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        dt_clf.fit(features_train, labels_train)
        svm_clf.fit(features_train, labels_train)

        bestDT.append(dt_clf.best_estimator_)
        bestSVM.append(svm_clf.best_estimator_)

    # set_DT = list(set([str(i) for i in bestDT]))   
    temp_DT = [str(i) for i in bestDT] 
    set_DT = list(set(temp_DT))
    temp_SVM = [str(i) for i in bestSVM] 
    set_SVM = list(set(temp_SVM))   

    selected_DT = sorted(zip(set_DT,[temp_DT.count(each) for each in set_DT]),key=lambda i:i[1],reverse=True)[0][0]
    selected_SVM = sorted(zip(set_SVM,[temp_SVM.count(each) for each in set_SVM]),key=lambda i:i[1],reverse=True)[0][0]

    for each in bestDT:
        if str(each) == selected_DT:
            DT = each
            break
    for each in bestSVM:
        if str(each) == selected_SVM:
            SVM = each
            break            
       
    # get the best parameters and use them only to run the algorithms
    # DT = dt_clf.best_estimator_
    # SVM = svm_clf.best_estimator_    
    GaussianNB = GaussianNB()
    classifier_type = [SVM, DT, GaussianNB]
    return classifier_type


#test algorithms and returns results
def analyze_feats(each_feature_set, my_dataset, classifier_type, scoresheet_highest_accuracy, 
                scoresheet_highest_precision, scoresheet_highest_recall):

    # run each type of classifier and return results
    try:
        total_results = []

        for index, each_clf in enumerate(classifier_type):
            results, feature_importances = test_classifier(each_clf, my_dataset, each_feature_set)
            if len(feature_importances) > 0:
                # results, feature_importances = test_classifier(each_clf, my_dataset, each_feature_set)
                print "####CLF NAME",each_clf
                print "#####Length of feature_importances",len(feature_importances)
                np.asarray(feature_importances)
                importances = zip(np.mean(feature_importances, axis=0),each_feature_set[1:])
                importances = sorted(importances,key=lambda i:i[0],reverse=True)
                print "#####Length of importances",len(importances)
                print importances

            print each_feature_set, results
            total_results.append(results)

        print "total_results",total_results 
        # for a given feature set, find the classifier with highest
        # precision/accuracy and store it in a list
        for index, num in enumerate(total_results):
            if num[1] == max([accuracy[1] for accuracy in total_results]):
                print "Highest accuracy: \t", num[0], num[1]
                scoresheet_highest_accuracy.append(
                    [each_feature_set, total_results[index]])
            if num[2] == max([precision[2] for precision in total_results]):
                print "Highest precision: \t", num[0], num[2]
                scoresheet_highest_precision.append(
                    [each_feature_set, total_results[index]])
            if num[3] == max([recall[3] for recall in total_results]):
                print "Highest recall: \t", num[0], num[3]
                scoresheet_highest_recall.append(
                    [each_feature_set, total_results[index]])
    except:
        pass


    return 

def best_results(scoresheet_highest_accuracy, scoresheet_highest_precision, scoresheet_highest_recall):
    # highest accuracy-feature set
    for index, num in enumerate(scoresheet_highest_accuracy):
        if num[1][1] == max([each[1][1] for each in scoresheet_highest_accuracy]):
            print "Best feature set by highest accuracy: \t"
            print num
            pass

    # highest precision-feature set
    for index, recall in enumerate(scoresheet_highest_recall):
        if recall[1][3] == max([each[1][3] for each in scoresheet_highest_recall]):
            print "Best feature set by highest recall: \t"
            print recall
            pass

    for index, prec in enumerate(scoresheet_highest_precision):
        if prec[1][2] == max([each[1][2] for each in scoresheet_highest_precision]):
            print "Best feature set by highest precision: \t"
            print prec
            return prec

    


