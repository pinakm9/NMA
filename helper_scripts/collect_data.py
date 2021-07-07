import os
import requests
import tarfile

HCP_DIR = '../data'
fnames = ["hcp_rest.tgz", "hcp_task.tgz"]
ids = ["bqp7m", "s4h8j"]
if not os.path.exists(HCP_DIR + '/' + fnames[1]):
    url = "https://osf.io/{}/download/".format(ids[1])
    response = requests.get(url, stream=True)
    file = tarfile.open(fileobj=response.raw, mode='r|gz')
    file.extractall(path=HCP_DIR)