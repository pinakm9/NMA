import os
import requests
import tarfile

# set up folders and dataset ids
HCP_DIR = '../data'
fnames = ["hcp_rest.tgz", "hcp_task.tgz", "hcp_rest.tgz"]
ids = ["bqp7m", "s4h8j", "g759t"]

# download and extract data
url = "https://osf.io/{}/download/".format(ids[2])
response = requests.get(url, stream=True)
file = tarfile.open(fileobj=response.raw, mode='r|gz')
file.extractall(path='.')