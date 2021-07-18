from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sklearn.decomposition 
import utility as ut 
import matplotlib.pyplot as plt

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
        clf = make_pipeline(StandardScaler(), SVC(kernel=self.kernel, **self.params))
        clf.fit(data, labels)
        if not hasattr(self, 'acc'):
            self.acc = clf.score(data, labels) * 100.0
        elif isinstance(self.acc, list):
            self.acc.append(clf.score(data, labels) * 100.0)
        else:
            self.acc = [self.acc]
            self.acc.append(clf.score(data, labels) * 100.0)
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