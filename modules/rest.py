import numpy as np

class Individual:
    """
    Description:
        Class for extracting data for an individual in the resting state data
    
    Attributes:
        db_path: path to HCP rest data
        subject_id: id of the subject 

    Methods:
        get_timeseries: fetches a resting state timeseries as indicated
        get_regions: fetches the brain region data that's common for all the subjects
    """

    def __init__(self, db_path, subject_id):
        self.db_path = db_path
        self.subject_id = subject_id

    def get_timeseries(self, index):
        """
        Description:
            fetches a resting state timeseries given an index
        
        Args:
            index: index of the timeseries required
        
        Returns:
            the requested timeseries
        """
        file_path = self.db_path + '/subjects/{}/timeseries/bold{}_Atlas_MSMAll_Glasser360Cortical.npy'.format(self.subject_id, index)
        return np.load(file_path)

    def get_regions(self):
        """
        Description: fetches the brain region data that's common for all the subjects

        Returns:
            the brain regions
        """
        file_path = self.db_path + '/regions.npy'
        return np.load(file_path)