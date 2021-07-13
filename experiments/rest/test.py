# add modules folder to Python's search path
from os import times
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')

import rest
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
import numpy as np
import os

# a cursory look at the data
db_path = '{}/data/hcp_rest'.format(module_dir)
subject_id = 15
timeseries_index = 2
subject = rest.Individual(db_path, subject_id) 
timeseries = subject.get_timeseries(timeseries_index)
#print(timeseries.shape)
regions = subject.get_regions()
subject_folder = 'subject_{}'.format(subject_id)
if not os.path.isdir(subject_folder):
    os.mkdir(subject_folder)
#print(regions, regions.shape)

# PCA on the timeseries to estimate the number of components
#"""
pca = PCA(n_components=200)
sources = pca.fit_transform(timeseries.T)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.scatter(list(range(1, 201)), np.cumsum(pca.explained_variance_ratio_))
ax.set_xlabel('principal components')
ax.set_ylabel('cumulative variance explained')
plt.savefig('{}/scree_plot_ts_{}.png'.format(subject_folder, timeseries_index)) 
plt.close(fig)
#"""
# a hasty ICA attempt on the timeseries
ica = FastICA(n_components=50, max_iter=2000)
sources = ica.fit_transform(timeseries.T)
print(sources.shape)


"""
data = np.random.rand(301) - 0.5
ps = np.abs(np.fft.fft(data))**2

time_step = 1 / 30
freqs = np.fft.fftfreq(data.size, time_step)
idx = np.argsort(freqs)

plt.plot(freqs[idx], ps[idx])
"""