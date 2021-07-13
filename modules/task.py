import numpy as np 

class Individual:
    """
    Class for extracting data for an individual in the resting state data
    """
    def __init__(self, db_path, subject_id):
        self.db_path = db_path
        self.subject_id = subject_id

    def get_timeseries(self, index):
        file_path = self.db_path + '/subjects/{}/timeseries/bold{}_Atlas_MSMAll_Glasser360Cortical.npy'.format(self.subject_id, index)
        return np.load(file_path)

    def get_regions(self):
        file_path = self.db_path + '/regions.npy'
        return np.load(file_path)

       