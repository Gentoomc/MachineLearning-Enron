#!/usr/bin/python


import pickle

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV



pca_feature_list = [
 'salary',
 'to_messages',
 'total_payments',
 'bonus',
 'restricted_stock',
 'shared_receipt_with_poi',
 'total_stock_value',
 'expenses',
 'from_messages',
 'from_this_person_to_poi',
 'from_poi_to_this_person']


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
### Remove the large outlier
data_dict.pop('TOTAL') 

my_dataset = data_dict

### create new features and save them in my_dataset ###
data = featureFormat(my_dataset, pca_feature_list,remove_NaN=True, remove_all_zeroes=False, remove_any_zeroes=False, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
new_features = pca.fit_transform(features)

for index, each_person in enumerate(my_dataset.values()):
	each_person['first_pc'] = new_features[index][0]
	each_person['second_pc'] = new_features[index][1]


################### my code: types of classifiers
from sklearn.naive_bayes import GaussianNB
GaussianNB = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.
from sklearn import svm
SVM = svm.SVC(C=1)
from sklearn import tree
DT = tree.DecisionTreeClassifier(min_samples_split=40)


classifier_type = [SVM, DT, GaussianNB]
feature_list = ['poi', 'first_pc', 'second_pc']

total_results = []
for each_clf in classifier_type:
	results = test_classifier(each_clf, my_dataset, feature_list)
	print results


################### run later
#dump_classifier_and_data(clf, my_dataset, features_list)

