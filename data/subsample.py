import os
import sys
import subprocess
import numpy as np
np.random.seed(0)

root = sys.argv[1]
divide = False 
if divide:
    subsample_root_tr=root+'_subset_tr'
    subsample_root_va=root+'_subset_va'
    os.makedirs(subsample_root_tr, exist_ok=True)
    os.makedirs(subsample_root_va, exist_ok=True)
else:
    subsample_root_va=root+'_subset'
    os.makedirs(subsample_root_va, exist_ok=True)

for i in os.listdir(root):
    if os.path.isdir(os.path.join(root, i)):
        if np.random.rand() < 0.05:
           subprocess.run("cp -r %s %s"%(os.path.join(root, i), subsample_root_va), shell=True)
        elif divide:
           subprocess.run("cp -r %s %s"%(os.path.join(root, i), subsample_root_tr), shell=True)
