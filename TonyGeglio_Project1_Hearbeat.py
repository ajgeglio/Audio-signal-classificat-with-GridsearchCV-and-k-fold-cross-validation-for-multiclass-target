
# model selection and metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import make_scorer, roc_auc_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
# keep track of time
from timeit import default_timer as stopwatch
import time
# For audio files 
import librosa
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import argparse
import os
import re
########################## Performance Functions ##################################

def calc_specificity(y_actual, y_pred):
    # calculates specificity
    return sum((y_pred < thresh) & (y_actual == 0)) / sum(y_actual == 0)

def print_report(y_actual, y_pred, y_pred_probas, thresh):
    # auc = roc_auc_score(y_actual, y_pred, multi_class='ovr')
    accuracy = metrics.accuracy_score(y_actual, y_pred)
    recall = metrics.recall_score(y_actual, y_pred, average='weighted')
    precision = metrics.precision_score(y_actual, y_pred, average = 'weighted')
    report = classification_report(y_actual, y_pred, target_names = classes_)
    # specificity = calc_specificity(y_actual, y_pred, thresh)
    # print('AUC:%.3f' % auc)
    print('accuracy:%.3f' % accuracy)
    print('recall:%.3f' % recall)
    print('precision:%.3f' % precision)
    print('Classification Report\n', report)
    # print('specificity:%.3f' % specificity)
    print(' ')
    return accuracy, recall, precision, report #, specificity

def report_performance(clf, X_train, X_test, y_train, y_test, thresh=0.5, clf_name="CLF"):
    print("[x] performance for {} classifier".format(clf_name))
    y_train_probas = clf.predict_proba(X_train)
    y_train_pred = np.argmax(y_train_probas, axis=1)
    y_test_probas = clf.predict_proba(X_test)
    y_test_pred = np.argmax(y_test_probas, axis=1)

    print('Training:')
    # train_auc, 
    train_accuracy, train_recall, train_precision, train_report = print_report( y_train, 
                                                                                y_train_pred, y_train_probas,
                                                                                thresh)
    print('Test:')
    # test_auc, 
    test_accuracy, test_recall, test_precision, test_report = print_report( y_test, 
                                                                            y_test_pred, y_test_probas,
                                                                            thresh)
    
    return {"train": {
                        # "auc": train_auc, 
                        "acc": train_accuracy, "recall": train_recall, "precision": train_precision,
                        "classification report": train_report
                        # "specificity": train_specificity
                        },
            "test": {
                        # "auc": test_auc, 
                        "acc": test_accuracy, "recall": test_recall, "precision": test_precision,
                        "classification report": test_report
                        # "specificity": test_specificity}
                        }
            }

########################### Get Files #############################################
def list_samples_labels(dir_):
    snd_files = []
    labels = []
    for dirs, subdirs, files in os.walk(dir_):
        snd_files.extend(os.path.join(dirs, x) for x in files if x.endswith(".wav"))
    for l in snd_files:
        labels.append(re.findall(r"\\([a-z]+)_\w+", l)[-1])
    print(len(labels)," labels counted")
    print(len(snd_files)," audio files counted")
    return snd_files, labels


##################### Training Definitions ##########################################

def train_svm(X_train, X_test, y_train, y_test, n_split=5):
    from sklearn import svm
    clf = svm.SVC(probability=True, max_iter=10000)
    warnings.filterwarnings("ignore")
    parameters =  {'C': [0.1, 1, 10],
                  'gamma': [0.01, 0.1, 10],
                  'kernel': ['linear', 'rbf', 'sigmoid']
                  }

    # auc_scoring = make_scorer(roc_auc_score)
    f1_scoring = make_scorer(f1_score, average='weighted')
    if n_split == 1:
        grid_clf = GridSearchCV(estimator=clf, param_grid = parameters, cv=[(slice(None), slice(None))],
                                scoring=f1_scoring, verbose=0)
    else:
        grid_clf = GridSearchCV(estimator=clf, param_grid = parameters, cv=n_split, scoring = f1_scoring, verbose=1)
    grid_clf.fit(X_train, y_train)
    print(grid_clf.best_estimator_)
    print(grid_clf.best_params_)

    performance_ = report_performance(grid_clf.best_estimator_, X_train, X_test, y_train, y_test, clf_name="SVM")
    return grid_clf.best_estimator_, grid_clf.best_params_, performance_

def train_gradient_boosting(X_train, X_test, y_train, y_test, n_split=1):
    from sklearn.ensemble import GradientBoostingClassifier
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=42)
    parameters = {  'n_estimators' : [5, 50, 100],
                    'max_depth' : [1, 5, 10],
                    'learning_rate' : [0.001, 0.01, 0.1]}
    
    # auc_scoring = make_scorer(roc_auc_score, average = 'weighted', multi_class='ovr')
    f1_scoring = make_scorer(f1_score, average='weighted')


    if n_split == 1:
        grid_clf = GridSearchCV(estimator=gbc, param_grid=parameters, cv=[(slice(None), slice(None))],
                                scoring = f1_scoring, verbose=1)
    else:
        grid_clf = GridSearchCV(estimator=gbc, param_grid=parameters, cv=n_split, scoring=f1_scoring, verbose=1)
    grid_clf.fit(X_train, y_train)

    print(grid_clf.best_estimator_)
    print(grid_clf.best_params_)
    gb_performance_ = report_performance(grid_clf.best_estimator_, X_train, X_test, y_train, y_test, clf_name="Gradient Boosting")
    return grid_clf.best_estimator_, grid_clf.best_params_, gb_performance_

def train_bayesian(X_train, X_test, y_train, y_test, n_split=5):
    from sklearn.naive_bayes import GaussianNB
    bayes = GaussianNB()
    parameters = {"priors": [None]}
    f1_scoring = make_scorer(f1_score, average='weighted')


    if n_split == 1:
        grid_clf = GridSearchCV(estimator=bayes, param_grid=parameters, cv=[(slice(None), slice(None))],
                                scoring = f1_scoring, verbose=1)
    else:
        grid_clf = GridSearchCV(estimator=bayes, param_grid=parameters, cv=n_split, scoring=f1_scoring, verbose=1)
    grid_clf.fit(X_train, y_train)
    print(grid_clf.best_estimator_)
    print(grid_clf.best_params_)
    nb_performance_ = report_performance(grid_clf, X_train, X_test, y_train, y_test, clf_name="GaussianNB")
    return nb_performance_

def train_rf(X_train, X_test, y_train, y_test, n_split=5):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(max_depth=6, random_state=42)
    # warnings.filterwarnings('ignore')
    warnings.filterwarnings('always') 
    rf.fit(X_train, y_train)

    n_estimators = (1,3,5,10,15)
    max_features = ['sqrt']
    max_depth = range(1, 5, 1)
    min_samples_split = range(2, 6, 2)
    criterion = ['gini', 'entropy']
    parameters = {'n_estimators': n_estimators, 'max_features': max_features,
                  'max_depth': max_depth, 'min_samples_split': min_samples_split, 'criterion': criterion}
    # auc_scoring = make_scorer(roc_auc_score, multi_class='ovr')
    f1_scoring = make_scorer(f1_score, average='weighted')

    if n_split == 1:
        grid_clf = GridSearchCV(estimator=rf, param_grid=parameters, cv=[(slice(None), slice(None))],
                                scoring=f1_scoring, verbose=0, n_jobs=-1)
    else:
        grid_clf = GridSearchCV(estimator=rf, param_grid=parameters, cv=n_split, scoring=f1_scoring, verbose=0,
                                n_jobs=-1)
    grid_clf.fit(X_train, y_train)

    print(grid_clf.best_estimator_)
    print(grid_clf.best_params_)
    performance_ = report_performance(grid_clf.best_estimator_, X_train, X_test, y_train, y_test, clf_name="RF")
    return grid_clf.best_estimator_, grid_clf.best_params_, performance_

def train_mlp(X_train, X_test, y_train, y_test, n_split=5):
    from sklearn.neural_network import MLPClassifier
    # warnings.filterwarnings("ignore")
    warnings.filterwarnings('always') 
    mlp = MLPClassifier(max_iter=500)

    activation = ['identity', 'logistic', 'tanh', 'relu']
    hidden_layer_sizes = [(10, 5), (20, 10), (40, 20)]
    alpha = [0.01, 0.1, 1, 10]
    parameters = {'activation': activation, 
                  'hidden_layer_sizes': hidden_layer_sizes, 'alpha': alpha}
    auc_scoring = make_scorer(roc_auc_score, multi_class = 'ovr')
    f1_scoring = make_scorer(f1_score, average='weighted')
    grid_clf = GridSearchCV(estimator=mlp, param_grid=parameters, cv=n_split, 
                            scoring=f1_scoring, verbose=0, n_jobs=-1)
    grid_clf.fit(X_train, y_train)

    print(grid_clf.best_estimator_)
    print(grid_clf.best_params_)
    performance_ = report_performance(grid_clf.best_estimator_, X_train, X_test, 
                                      y_train, y_test, clf_name="MLP")
    return grid_clf.best_estimator_, grid_clf.best_params_, performance_

##################### Getting features out of audio files ##############################################
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def scale_data():
    scaler = StandardScaler()
    if i == 0:
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        i+=1
    else: pass
    return X_train, X_test

# Only needed when classifier does not handle multiclass
def binarize_labels(labels):
    lb = preprocessing.LabelBinarizer()
    return lb.fit_transform(labels), lb.classes_
# bin_labels, classes = binarize_labels(labels_)

def reshape_features():
    nsamples, nx, ny = features.shape
    reshape_features = features.reshape((nsamples,nx*ny))
    print(reshape_features.shape)

def label_encoder(labels):
    label2id, id2label = dict(), dict()
    for i, label in enumerate(np.unique(labels)):
        label2id[label] = i
        id2label[i] = label
    return label2id, id2label

if __name__ == "__main__":
    start_time = stopwatch()

    parser = argparse.ArgumentParser(description='Creates a dataset of taps from a wavfile of recorded typed sentences and associated neonode file')
    parser.add_argument('--dir', help="directory where the heart sounds are", dest="dir_", default='G:\My Drive\_2023_Spring\SAT5114\Statistical_ML\Project\heartbeat_data\data')
    parser.add_argument('--plot', help="plot the performance", action="store_true")
    parser.add_argument('--train', help="train the classifier", action="store_true")
    args = parser.parse_args()

    dir_ = args.dir_
    # label_files, tap_files = list_samples_labels(dir_)
    trn_file_paths, labels_ = list_samples_labels(dir_)
    label2id, id2label = label_encoder(labels_)
    enc_labels = [label2id[lbl] for lbl in labels_]
    classes_ = np.unique(labels_)
    print("class labels", classes_)


    if 'heartbeat_features.csv' in os.listdir(r'./heartbeat_data/data/'):
        features = np.genfromtxt(r'./heartbeat_data/data/heartbeat_features.csv', delimiter=',')
        print("the features were loaded: ", features.shape)
    
    else: 
        print("extracting features from audio files... ...")
        features = [extract_features(f) for f in trn_file_paths]
        features = np.array(features)
        np.savetxt('.\hearbeat_data\data\heartbeat_features.csv', features, delimiter=',', newline='\n', encoding=None)
        print("n_samples x n_features created: ", features.shape)
    

    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                        enc_labels,
                                                        stratify=enc_labels, 
                                                        test_size=0.25,
                                                        random_state=109) 
    
    if args.train:
        # print("SVM Classifier Reporting\n")
        # best_estimator_, best_params_, svm_performance_ = train_svm(X_train, X_test, y_train, y_test, n_split=5)
        print("Naive Bays")
        best_params_, nb_performance_ = train_bayesian(X_train, X_test, y_train, y_test, n_split=5)
        # print('Random Forest')
        # best_estimator_, best_params_, rf_performance_ = train_rf(X_train, X_test, y_train, y_test, n_split=5)
        # print('MLP')
        # best_estimator_, best_params_, mlp_performance_ = train_mlp(X_train, X_test, y_train, y_test, n_split=5)
        # print('Gradient Boosing')
        # best_estimator_, best_params_, gb_performance_ = train_gradient_boosting(X_train, X_test, y_train, y_test, n_split=5)

    ################## Performance Summary Plot ##########################################
    if args.plot:
        f1_train_scores = [0.69, 0.74, 0.55, 0.77, 0.75]
        f1_test_scores = [0.62, 0.65, 0.53, 0.69, 0.64]
        # performances = [svm_performance_, rf_performance_, nb_performance_, 
        #                 mlp_performance_, gb_performance_]
        models = ["SVM", "Random Forests","GaussianNB",
                  "Multi-layer perceptron", "Gradient Boosting"]
        fig, ax = plt.subplots( figsize = (7,4), tight_layout=False)
        i=0
        # for r in range(2):
        #     for c in range(3):
        # ax.bar(height = performances[i]['train'].values(), 
        #             x = list(performances[i]['train'].keys()),
        #             width = -0.25,
        #             align = 'edge',
        #             label = "training"
        #             )
        ax.barh(width = f1_train_scores, 
                    y = models,
                    height = -0.25,
                    align = 'edge',
                    label = "training"
                    )
        ax.barh(width = f1_test_scores, 
                    y = models,
                    height = 0.25,
                    align = 'edge',
                    label = "test"
                    )
        ax.set_xlim(0.4,1)
        # ax.set_title(f"{models[i]}")
        i+=1
        plt.suptitle('Statistical Classifiers Weighted F1 Score')
        plt.legend()
        plt.savefig('performance.png')
 