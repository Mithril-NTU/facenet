import os
import sys
import subprocess
import numpy as np
np.random.seed(0)

root=sys.argv[1]
subsample_root=root+'_subset'
os.makedirs(subsample_root, exist_ok=True)
for i in os.listdir(root):
    if os.path.isdir(os.path.join(root, i)):
        if np.random.rand() < 0.02:
           subprocess.run("cp -r %s %s"%(os.path.join(root, i), subsample_root), shell=True)
