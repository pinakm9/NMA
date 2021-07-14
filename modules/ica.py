import numpy as np
from sklearn.mixture import GMM

class ICA:
    """
    Description:
        Class to perform ICA on steriods
        
    Attributes:
        data: space-time matrix
    
    Methods:
        model_order_selection: do PCA in space to explain variance with respect to eigenspectrum
        z_transform: Z-transforms ICA components
        fit_GMM: fits a Gaussian mixture model to the z-transformed components



        
To do: determine if a component is biological or not 
fit 2 gaussian GLM like model to find the fitting curves for noise and signal
Find ROIs that are more likely to be in the signal curve (P<0.05)
(Plotting it back on a brain?)

    """

    def __init__(self, data):
        self.data = data


    def model_order_selection(self, n_components): 
        """
        Description:
            Model order selection: estimation of how many components in the data are relevant
            1. do PCA in space to explain variance with respect to eigenspectrum for the data
            2. Generate random (Gaussian) data of the same size and do PCA on it as well
            3. the number of components = the point where the explained variances meet.

        Arguments:
            n_components: number of components the PCA will keep.
        
        Returns:
            arrays of explained variances in the data and the noise + a plot of them
            estimated number of components that explain most of the variance in the data

        """

        pca = PCA(n_components=200)
        sources = pca.fit_transform(timeseries.T)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.scatter(list(range(1, 201)), np.cumsum(pca.explained_variance_ratio_))
        ax.set_xlabel('principal components')
        ax.set_ylabel('cumulative variance explained')
        plt.savefig('{}/scree_plot_ts_{}.png'.format(subject_folder, timeseries_index)) 
        plt.close(fig)

        file_path = self.db_path + '/subjects/{}/timeseries/bold{}_Atlas_MSMAll_Glasser360Cortical.npy'.format(self.subject_id, index)
        return np.load(file_path)

    def get_regions(self):
        file_path = self.db_path + '/regions.npy'
        return np.load(file_path)


    def fit_GMM(self, ica_comps, n_gmm_comps=2):
        """
        Description:
            fits a Gaussian mixture model to the z-transformed components 
        
        Args:
            ica_comps: the ICA components to fit your Gaussian mixture models to
            n_gmm_comps: number of components in the mixture model
        
        Returns:
            the computed mixture models
        """
        gmms = []
        model = GMM(n_components=n_gmm_comps)
        for comp in ica_comps:
            gmms.append(model.fit(comp))
        return gmms

    def test_hyp(self, p_value):
        pass
     