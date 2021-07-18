# add modules folder to Python's search path
from os import times
import sys
from pathlib import Path
from os.path import dirname, realpath, abspath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')


# import remaining modules
import task
import methods 
import matplotlib.pyplot as plt

# set up conditions and collect list
db_path = '../../data/hcp_task'
group = task.Group(db_path)
conditions_0 = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools']
conditions_2 = ['2bk_body', '2bk_faces', '2bk_places', '2bk_tools']
X_0, Y_0 = group.extract_cons(conditions_0) 
X_2, Y_2 = group.extract_cons(conditions_2)

# set up methods
pca = methods.PCA()
svm_0 = methods.SVM(kernel='rbf', gamma=0.7, C=1.0)
svm_2 = methods.SVM(kernel='rbf', gamma=0.7, C=1.0)

# fit SVM on entire data
"""
svm = methods.SVM(kernel='rbf', gamma=0.7, C=1.0)
svm.fit(X, y)
print("Accuracy of SVM: {}".format(svm.acc))
""" 

# fit SVM on principal components
comp_list = list(range(2, 50))
for n_components in comp_list:
    pca.compute(X_0, n_components)
    svm_0.fit(pca.new_data, Y_0)
    pca.compute(X_2, n_components)
    svm_2.fit(pca.new_data, Y_2)


# plot accuarcies
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.plot(comp_list, svm_0.acc, c='b', label='0bk')
ax.scatter(comp_list, svm_0.acc, c='b')
ax.plot(comp_list, svm_2.acc, c='r', label='2bk')
ax.scatter(comp_list, svm_2.acc, c='r')
ax.set_xlabel("number of principal components")
ax.set_ylabel("accuracy (%)")
plt.legend()
plt.savefig('../plots/task_roi_pca_comps_vs_svm_acc.png')
