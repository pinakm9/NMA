import numpy as np

class ICA:
    """
    Description:
        Class to perform ICA on steriods
        
    Attributes:
        data: space-time matrix
    
    Methods:
        do_PCA: do PCA in space to explain variance with respect to eigenspectrum 

Generate random (Gaussian) data of the same size and do PCA on it with explain variance with respect to eigenspectrum
The number of components =  the point where the cumulative variance lines in 1 and 2 meet. Now we are done with model order selection!
Z-transform
(Plot the components as brain maps after doing z-transform ←-- easiest step???)
To do: determine if a component is biological or not 
fit 2 gaussian GLM like model to find the fitting curves for noise and signal
Find ROIs that are more likely to be in the signal curve (P<0.05)
(Plotting it back on a brain?)

    """

    def __init__(self, data):
        self.data = data


    def do_pca(self, n_components): 
        """
        Description:
            do PCA in space to explain variance with respect to eigenspectrum 

        Arguments:
            n_components: number of components the PCA will keep.
            two:
        
        Returns:
            array of cummulative variances explained and plots it

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


     