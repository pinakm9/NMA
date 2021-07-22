from random import random
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sklearn.decomposition 
import utility as ut 
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import optunity
import optunity.metrics

class PCA:
    """
    Personalized class for PCA

    Attributes:
        n_components: number of components used in during last compute call
        new_data: reduced data generated during last compute call

    Methods:
        compute: computes required principal components

    """
    def __init__(self):
        pass

    @ut.timer
    def compute(self, data, n_components):
        """
        Description:
            computes required principal components

        Args:
            data: data matrix of shape (samples, features)
            n_components: number of principal components required

        Returns:
            a PCA object containing the components, variance explained data
        """
        self.n_components = n_components
        pca = sklearn.decomposition.PCA(n_components=n_components)
        self.new_data = pca.fit_transform(data)
        return pca


    @ut.timer
    def per_individual(self, data, n_components):
        """
        Description:
            performs PCA per individual 
        """
        new_data = np.zeros((data.shape[0], data.shape[2], n_components))
        pca = sklearn.decomposition.PCA(n_components=n_components)
        for i, subject in enumerate(data):
            new_data[i] = pca.fit_transform(subject.T)
        return new_data


    @ut.timer
    def lenca(self, data, n_components):
        """
        Description:
            performs PCA after epiphany
        """
        n_subjects, roi_dim, time_dim = data.shape
        data = data.reshape(roi_dim * 4, -1)
        
        """
        for i, row in enumerate(data):
            std = np.std(row)
            mean = np.mean(row)
            data[i, :] = (row - mean) / std 
        #data = self.normalize(data)
        """
        
        pca = sklearn.decomposition.PCA(n_components=n_components)
        new_labels = []
        for i in range(4):
            new_labels += [i] * roi_dim
        new_data = pca.fit_transform(data)
        return new_data, np.array(new_labels), pca.explained_variance_ratio_#pca.fit_transform()



    @ut.timer
    def pca_roi(self, data, labels, n_components):
        """
        Description:
            PCA on samples = subjects x time, features = ROIs
        """
        _, _, time_dim = data.shape
        data = data.reshape(data.shape[0] * data.shape[2], -1)
        #data = self.normalize(data)
        pca = sklearn.decomposition.PCA(n_components=n_components)
        new_labels = []
        for label in labels:
            new_labels += [label] * time_dim
        new_data = pca.fit_transform(data)
        return new_data, np.array(new_labels),  pca.explained_variance_ratio_


    def regular(self, data, n_components):
        """
        Description:
            Vanilla PCA without any mods
        """
        pca = sklearn.decomposition.PCA(n_components=n_components)
        return pca.fit_transform(data),  pca.explained_variance_ratio_


    @ut.timer
    def normalize(self, data, norm_class=MinMaxScaler):
        """
        Description:
            normalizes data according to provided normalizer class
        
        Args:
            data: data to be normalized
            norm_class: a Python class that normalizes data, default = MinMaxScaler

        Returns:
            normalized data
        """
        scaler = norm_class()
        return scaler.fit_transform(data)


class SVM:
    """
    Description:
        Personalized class for support vector machines 

    Attributes:
        kernel: SVM kernel to be used
        params: extra parameters of sklearn SVC class
        acc: accuracy score, available after fitting

    Methods:
        fit: fits SVM to labelled date
    """

    def __init__(self, kernel='rbf', **params):
        self.kernel = kernel
        self.params = params
        self.scores = []

    @ut.timer
    def fit(self, data, labels):
        """
        Description: 
            fits SVM to labelled data

        Args:
            data: data to be classified
            labels: corresponding labels

        Returns:
            Pipeline object from sklearn containg SVM-fitting data
        """
        clf = SVC(kernel=self.kernel, **self.params)
        clf.fit(data, labels)
        self.scores.append(clf.score(data, labels))#clf.score(data, labels))
        return clf 

    @ut.timer
    def cross_val(self, data, labels, k_folds=8):
        """
        Description: 
            fits and cross-validates SVM to labelled data

        Args:
            data: data to be classified
            labels: corresponding labels
            k_folds: number of folds during cross validation

        Returns:
            Pipeline object from sklearn containg SVM-fitting data
        """
        clf = SVC(kernel=self.kernel, **self.params)
        scores = cross_val_score(clf, data, labels, cv=k_folds)
        self.scores.append(scores)#clf.score(data, labels))
        return clf 


    @ut.timer
    def massive_shuffle(self, data, labels, n_shuffle=100):
        """
        Description:
            Shuffle your data to death
        """
        for _ in range(n_shuffle):
            data, labels = shuffle(data, labels, random_state = np.random.randint(int(1e6), int(1e9)))
        return data, labels 


    @ut.timer
    def cross_val_diy(self, data, labels, k_folds=8, n_shuffle=100):
        """
        """
        data, labels = self.massive_shuffle(data, labels, n_shuffle)
        kf = KFold(n_splits=k_folds)
        
        for train_index, test_index in kf.split(data):
            clf = SVC(kernel=self.kernel, **self.params)
            clf.fit(data[train_index], labels[train_index])
            score = clf.predict(data[test_index])
            print(score) 
            print("test", labels[test_index])
            print("train", labels[train_index][:100])
            self.scores.append(clf.score(data[test_index], labels[test_index]))
        return clf


    def plot_acc(self, x_data, file_path):
        """
        Description:
            plots accuracies
        
        Args:
            x_data: x values corresponding to accuracies
            file_path: path for saving the plot
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.plot(x_data, self.acc)
        ax.scatter(x_data, self.acc)
        ax.set_ylabel("accuracy")
        plt.savefig(file_path)


    def tune(self, data, labels):
        # score function: twice iterated 10-fold cross-validated accuracy
        @optunity.cross_validated(x=data, y=labels, num_folds=10, num_iter=2)
        def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
            model = sklearn.svm.SVC(C=10 ** logC, gamma=10 ** logGamma).fit(x_train, y_train)
            decision_values = model.decision_function(x_test)
            return optunity.metrics.roc_auc(y_test, decision_values)

        # perform tuning
        hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])

        # train model on the full training set with tuned hyperparameters
        optimal_model = sklearn.svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(data, labels)
        return optimal_model



