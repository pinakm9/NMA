from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import utility as ut 

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
        self.acc = clf.score(data, labels)
        return clf