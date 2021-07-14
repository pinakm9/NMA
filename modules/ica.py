import numpy as np
from sklearn.mixture import GaussianMixture
import scipy.stats as ss

class ICA:
    """
    Description:
        Class to perform ICA on steriods
        
    Attributes:
        data: space-time matrix
    
    Methods:
        model_order_selection: do PCA in space to estimate necessary numbers of components
        z_transform: Z-transforms ICA components
        fit_GMM: fits a Gaussian mixture model to the z-transformed components




To do: determine if a component is biological or not 
fit 2 gaussian GLM like model to find the fitting curves for noise and signal
Find ROIs that are more likely to be in the signal curve (P<0.05)
(Plotting it back on a brain?)

    """

    def __init__(self, data):
        self.data = data


    def model_order_selection(self, n_components, threshold): 
        """
        Description:
            Model order selection: estimation of how many components in the data are relevant
            1. do PCA in space to explain variance with respect to eigenspectrum for the data
            2. plot and compare with threshold
            3. the number of components = the point where the explained variances meet.

        Arguments:
            n_components: number of components the PCA will keep
            threshold: total percentage of variance explained by necessary components

        Returns:
            arrays of explained variances in the data and the noise + a plot of them
            estimated number of components that explain most of the variance in the data

        """
        # PCA on the timeseries to estimate the number of components needed to explain 90% of variance
        pca = PCA(n_components=200)
        pca.fit_transform(self.data.T)
        variance_data = np.cumsum(pca.explained_variance_ratio_)

        # scree plot 
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.scatter(list(range(1, n_components+1)), variance_data)
        ax.scatter(list(range(1, n_components+1)), np.full(200, threshold)) # threshold line at 90% variance explained
        ax.set_xlabel('principal components')
        ax.set_ylabel('cumulative variance explained')
        # plt.savefig('{}/scree_plot_ts_{}.png'.format(subject_folder, timeseries_index)) Where do we want to save the figure?
        plt.close(fig)
        
        # calculate estimated necessary number of components
        diff = threshold - variance_data
        n_comp = [i for i, x in enumerate(diff) if x > 0][-1]
        
        return n_comp
        

       





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
        model = GaussianMixture(n_components=n_gmm_comps)
        for comp in ica_comps:
            gmm = model.fit(comp.reshape(-1, 1))
            # order the gmm components according to ascending means
            asc_order = np.argsort(gmm.weights_)
            gmm.weights_ = gmm.weights_[asc_order]
            gmm.means_ = gmm.means_[asc_order]
            gmm.covariances_ = gmm.covariances_[asc_order]
            gmms.append(gmm)
        return gmms

    def select_ROIs(self, ica_comps, gmms, threshold):
        """
        Description:
            Selects ROIs based on if it belongs to the signal or noise part of an independent component 
            and keeps/removes them in/from the component accordingly

        Args:
            ica_comps: independent components
            gmm: GaussianMixture objects in skleran fitted to the histograms of the independent components
            threshold: p-value determing the range on standard Gaussian distribution in which the contribution
                of an ROI must lie in order to be selected 
        """
        a = -ss.norm.ppf(threshold/2.0)
        n_comps, n_ROIs = ica_comps.shape 
        for i in range(n_comps):
            loc = gmms[i].weights_[-1]
            scale = np.sqrt(gmms[i].covariances_[-1][0][0])
            for roi in range(n_ROIs):
                z = (ica_comps[i][roi] - loc) / scale
                if z > a or z < -a:
                    ica_comps[i][roi] = 0.0