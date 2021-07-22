import numpy as np
from numpy.core import einsumfunc
import utility as ut
from sklearn import preprocessing

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
        load_evs_con: load EVs (explanatory variables) data for one condition of task experiment and creates timeseries
        get_regions: fetches the brain region data that's common for all the subjects
    """

    def __init__(self, db_path, subject_id):
        self.db_path = db_path
        self.subject_id = subject_id
        self.exp = 'WM'
        self.experiments = {
                'MOTOR'      : {'runs': [5,6],   'cond':['lf','rf','lh','rh','t','cue']},
                'WM'         : {'runs': [7,8],   'cond':['0bk_body','0bk_faces','0bk_places','0bk_tools','2bk_body','2bk_faces','2bk_places','2bk_tools']},
                'EMOTION'    : {'runs': [9,10],  'cond':['fear','neut']},
                'GAMBLING'   : {'runs': [11,12], 'cond':['loss','win']},
                'LANGUAGE'   : {'runs': [13,14], 'cond':['math','story']},
                'RELATIONAL' : {'runs': [15,16], 'cond':['match','relation']},
                'SOCIAL'     : {'runs': [17,18], 'cond':['mental','rnd']}
            }

    def load_single_timeseries(self, run, remove_mean=True):
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



    def load_evs(self):
        """
        Description:
            load EVs (explanatory variables) data for one task experiment

        Args:
            subject (int): 0-based subject ID to load

        Returns
            evs (list of lists): A list of frames associated with each condition

        """

        TR = 0.72  # Time resolution, in seconds
        frames_list = []
        for run in ['_RL','_LR']:
            task_key = 'tfMRI_'+self.exp+run
            for cond in self.experiments[self.exp]['cond']:
                file_path = self.db_path + '/subjects/{}/EVs/{}/{}.txt'.format(self.subject_id, task_key, cond)
                ev_array = np.loadtxt(file_path, ndmin=2, unpack=True)
                ev       = dict(zip(["onset", "duration", "amplitude"], ev_array))
                # Determine when trial starts, rounded down
                start = np.floor(ev["onset"] / TR).astype(int)
                # Use trial duration to determine how many frames to include for trial
                duration = np.ceil(ev["duration"] / TR).astype(int)
                # Take the range of frames that correspond to this specific trial
                frames = [s + np.arange(0, d) for s, d in zip(start, duration)]
                frames_list.append(frames)
        return frames_list




    def load_evs_con(self, condition, remove_mean=False):
        """
        Description:
            load EVs (explanatory variables) data for one condition of task experiment and creates timeseries

        Args:
            subject (int): 0-based subject ID to load
            condition: '0bk_body','0bk_faces','0bk_places','0bk_tools','2bk_body','2bk_faces','2bk_places','2bk_tools'
            remove_mean: indicator for removing mean

        Returns
            A timeseries for all ROIs (360,78)

        """
        TR = 0.72  # Time resolution, in seconds
        list_runs = [7,8]
        ts = []
        for i, run in enumerate(['_RL','_LR']):
            task_key = 'tfMRI_'+self.exp+run
            file_path = self.db_path + '/subjects/{}/EVs/{}/{}.txt'.format(self.subject_id, task_key, condition)
            ev_array = np.loadtxt(file_path, ndmin=2, unpack=True)
            ev       = dict(zip(["onset", "duration", "amplitude"], ev_array))
            # Determine when trial starts, rounded down
            start = np.floor(ev["onset"] / TR).astype(int)
            # Use trial duration to determine how many frames to include for trial
            duration = np.ceil(ev["duration"] / TR).astype(int)
            # Take the range of frames that correspond to this specific trial
            frames = [s + np.arange(0, d) for s, d in zip(start, duration)]
            #frames_list.append(frames)
            self.experiments[self.exp]['runs']
            file_path = self.db_path + '/subjects/{}/timeseries/bold{}_Atlas_MSMAll_Glasser360Cortical.npy'.format(self.subject_id, list_runs[i])
            newts= np.load(file_path)[:, frames[0]]
            if remove_mean:
                newts -= newts.mean(axis=1, keepdims=True)
            ts.append(newts)
        return np.hstack(ts)



    def get_regions(self):
        """
        Description: fetches the brain region data that's common for all the subjects

        Returns:
            the brain regions
        """
        file_path = self.db_path + '/regions.npy'
        return np.load(file_path)

class Group:
    """
    Description:
        Class for extracting data for a condition for all subjects

    Attributes:
        db_path: path to HCP task data

    Methods:
        extractall:

    """
    def __init__(self, db_path):
        self.db_path = db_path
        self.n_subjects = 339

    @ut.timer
    def extract_con(self, condition):
        """
        Description:
            Extract all time series for all subjects for a specific condition

        Args:
            condition: '0bk_body','0bk_faces','0bk_places','0bk_tools','2bk_body','2bk_faces','2bk_places','2bk_tools'

        Returns:
            A list all timeseries for all ROIs (360,78) and all subjects

        """
        X = []
        for subject_id in range(self.n_subjects):
            subject = Individual(self.db_path, subject_id)
            X.append(subject.load_evs_con(condition))
        return X

    @ut.timer
    def extract_cons(self, conditions):
        """
        Description:
            Extract all time series for all subjects for a list of conditions

        Args:
            conditions: subset of {'0bk_body','0bk_faces','0bk_places','0bk_tools','2bk_body','2bk_faces','2bk_places','2bk_tools'}

        Returns:
            features (flattened) and labels for all subjects

        """
        data, labels = [], [] 
        for label, condition in enumerate(conditions):
            data += self.extract_con(condition)
            labels += [label] * self.n_subjects
        return np.array(data), np.array(labels) 


    @ut.timer
    def normalize_individuals(self, data):
        """
        Description:
            Normalizes data for a single individual
        """
        for i, subject in enumerate(data):
            data[i, :, :] =  (preprocessing.StandardScaler().fit_transform(subject.T)).T
        return data


    @ut.timer
    def normalize_across_cons(self, data):
        """
        Description:
            Normalizes data for a single individual
        """
        for i, subject in enumerate(data):
            data[i, :, :] =  (preprocessing.StandardScaler().fit_transform(subject))
        return data


    @ut.timer
    def squash_frames(self, data):
        """
        Description:
            squashes the frames into average
        """
        new_data = np.zeros((data.shape[0], data.shape[1]))
        for i, subject in enumerate(data):
            for j, roi in enumerate(subject):
                new_data[i, j] = np.mean(roi)
   
        return new_data
