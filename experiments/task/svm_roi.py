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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# collect data
db_path = '../../data/hcp_task'
group = task.Group(db_path)
conditions = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools']
X, y = group.extract_cons(conditions)

# fit SVM
clf = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
clf.fit(X, y)