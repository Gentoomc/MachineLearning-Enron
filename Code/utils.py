from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import tree
from tester import test_classifier

# for each feature set, run classifiers and return results
# master_list_features


def analyze_feats(each_feature_set, my_dataset, scoresheet_highest_accuracy,
                  scoresheet_highest_precision):
    data = featureFormat(my_dataset, each_feature_set, sort_keys=True)
    labels, features = targetFeatureSplit(data) 
    features_train, features_test, labels_train, labels_test = (
        train_test_split(features, labels, test_size=0.5, random_state=42))

    # ################## For each feature set, tune the SVC parameter and
    # return the best SVC parameters
    # tuned_parameters = [{'kernel': ['rbf'], 'C': [1, 3, 10, 100, 1000],
    # 'degree':[1,2,3]}]
    # #score = 'precision'
    # clf = GridSearchCV(SVM, tuned_parameters)
    # clf.fit(features_train, labels_train)
    # SVM = clf.best_estimator_
    # print SVM

    # For each feature set, tune the SVC parameter and return the best SVC
    # parameters
    DT_tuned_parameters = [{'min_samples_split': [30, 40, 50]}]
    # score = 'precision'
    dt_clf = GridSearchCV(tree.DecisionTreeClassifier(), DT_tuned_parameters)
    dt_clf.fit(features_train, labels_train)
    DT = dt_clf.best_estimator_
    print DT
    classifier_type = [DT]
    # continue
    # run each type of classifier and return results
    try:
        total_results = []
        for index, each_clf in enumerate(classifier_type):
            results = test_classifier(each_clf, my_dataset, each_feature_set)
            print each_feature_set, results
            total_results.append(results)

        # for a given feature set, find the classifier with highest
        # precision/accuracy and store it in a list
        for index, num in enumerate(total_results):
            if num[1] == max([accuracy[1] for accuracy in total_results]):
                # print "Highest accuracy: \t", num[0], num[1]
                scoresheet_highest_accuracy.append(
                    [each_feature_set, total_results[index]])
            if num[1] == max([precision[1] for precision in total_results]):
                # print "Highest precision: \t", num[0], num[1]
                scoresheet_highest_precision.append(
                    [each_feature_set, total_results[index]])
    except:
        pass
