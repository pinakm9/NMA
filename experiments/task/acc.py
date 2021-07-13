# add modules folder to Python's search path
from os import times
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')


# a cursory look at the data
db_path = '{}/data/hcp_task'.format(module_dir)
task = 'tfMRI_WM_LR'
cor_type = '0bk_cor'
err_type = '0bk_err'
subjects = []
for sub_id in range(339):
    cor_file = db_path + '/subjects/{}/EVs/{}/{}.txt'.format(sub_id, task, cor_type)
    err_file = db_path + '/subjects/{}/EVs/{}/{}.txt'.format(sub_id, task, err_type)
    with open(cor_file,'r') as file:
        cor_count = len(file.read().split('\n'))
    with open(err_file,'r') as file:
        err_count = len(file.read().split('\n'))
    subjects.append([[cor_count, err_count]]) 

task = 'tfMRI_WM_LR'
cor_type = '2bk_cor'
err_type = '2bk_err'
for sub_id in range(339):
    cor_file = db_path + '/subjects/{}/EVs/{}/{}.txt'.format(sub_id, task, cor_type)
    err_file = db_path + '/subjects/{}/EVs/{}/{}.txt'.format(sub_id, task, err_type)
    with open(cor_file,'r') as file:
        cor_count = len(file.read().split('\n'))
    with open(err_file,'r') as file:
        err_count = len(file.read().split('\n'))
    subjects[sub_id].append([cor_count, err_count]) 

#print(subjects)
for sub_id in range(339):
    print('Trial 1, Working memory, Subject #{}: 0bk_Correct: {}, 0bk_Incorrect: {}, 2bk_Correct: {}, 2bk_Incorrect: {}'.\
            format(sub_id, subjects[sub_id][0][0], subjects[sub_id][0][1], subjects[sub_id][1][0], subjects[sub_id][1][1])) 