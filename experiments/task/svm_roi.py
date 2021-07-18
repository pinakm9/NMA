# add modules folder to Python's search path
from os import times
import sys
from pathlib import Path
from os.path import dirname, realpath, abspath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')
#print(module_dir)
#print(script_dir) 

# import remaining modules
import task
import methods 


# collect data
db_path = '../../data/hcp_task'
group = task.Group(db_path)
conditions = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools']
X, y = group.extract_cons(conditions)
print(y, X.shape)
# fit SVM
svm = methods.SVM(kernel='rbf', gamma=0.7, C=1.0)
svm.fit(X, y)
print("Accuracy of SVM: {}".format(svm.acc))