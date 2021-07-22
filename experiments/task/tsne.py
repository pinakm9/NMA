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
from sklearn.manifold import TSNE

# set up conditions and collect list
db_path = '../../data/hcp_task'
group = task.Group(db_path)
conditions_0 = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools']
conditions_2 = ['2bk_body', '2bk_faces', '2bk_places', '2bk_tools']
X_0, Y_0 = group.extract_cons(conditions_0, remove_mean=True)
X_0 = group.squash_frames(X_0)
