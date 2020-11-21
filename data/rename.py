import os, sys

root = sys.argv[1]

dirs = [os.path.join(root, d) for d in os.listdir(root)]
for d in dirs:
    fs = [os.path.join(d, f) for f in os.listdir(d)]
    for i, f in enumerate(fs):
        d_basename = os.path.basename(d)
        f_ext = f.split('.')[-1]
        os.rename(f, os.path.join(d, '%s_%04d.%s'%(d_basename, i+1, f_ext)))

