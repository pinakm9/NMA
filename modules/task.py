import numpy as np

class Individual:
    """
    Description:
        Class for extracting data for an individual in the working memory task data
    
    Attributes:
        db_path: path to HCP task data
        subject_id: id of the subject 

    Methods:
        load_single_timeseries: load timeseries data for a single subject and single run
        load_evs: load EVs (explanatory variables) data for one task experiment
        get_regions: fetches the brain region data that's common for all the subjects
    """

    def __init__(self, db_path, subject_id):
        self.db_path = db_path
        self.subject_id = subject_id
        self.experiment = WM

    def load_single_timeseries(self, run, remove_mean=True)):
        """
        Description:
            load timeseries data for a single subject and single run
        
        Args:
            subject (int):      0-based subject ID to load
            run (int):          rund 7 or 8 for wm task
            remove_mean (bool): If True, subtract the parcel-wise mean (typically the mean BOLD signal is not of interest)
        
        Returns:
            ts (n_parcel x n_timepoint array): Array of BOLD data values
        """
        file_path = self.db_path + '/subjects/{}/timeseries/bold{}_Atlas_MSMAll_Glasser360Cortical.npy'.format(self.subject_id, run)
        ts= np.load(file_path)
        if remove_mean:
            ts -= ts.mean(axis=1, keepdims=True)
        return ts



    def load_evs(self, run):
        """
        Description:
            load EVs (explanatory variables) data for one task experiment

        Args:
            subject (int): 0-based subject ID to load
            run (int):          rund 7 or 8 for wm task

        Returns
            evs (list of lists): A list of frames associated with each condition

        """
    frames_list = []
    task_key = 'tfMRI_'+self.experiment+['RL','LR'][run]
    for cond in EXPERIMENTS[self.experiment]['cond']:    
        file_path = self.db_path + '/subjects/{}/EVs/{}/{}.txt'.format(self.subject_id, task_key, cond)
        ev_array = np.loadtxt(file_patch, ndmin=2, unpack=True)
        ev       = dict(zip(["onset", "duration", "amplitude"], ev_array))
        # Determine when trial starts, rounded down
        start = np.floor(ev["onset"] / TR).astype(int)
        # Use trial duration to determine how many frames to include for trial
        duration = np.ceil(ev["duration"] / TR).astype(int)
        # Take the range of frames that correspond to this specific trial
        frames = [s + np.arange(0, d) for s, d in zip(start, duration)]
        frames_list.append(frames)
    return frames_list

    def get_regions(self):
        """
        Description: fetches the brain region data that's common for all the subjects

        Returns:
            the brain regions
        """
        file_path = self.db_path + '/regions.npy'
        return np.load(file_path)

       