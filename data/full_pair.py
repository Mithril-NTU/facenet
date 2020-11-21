from pathlib import Path
import h5py
import os, sys
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, save_npz
import numpy as np
image_size = 160
def make(images_path, out_path):
    images_path = Path(images_path)
    #paths = sorted(images_path.glob('*/*'), key=str)
    #shape = (len(paths), image_size, image_size, 3)
    #data = h5py.File(out_path, 'w')
    #dataset = data.create_dataset('images', shape=shape)
    #for i, path in enumerate(paths):
    #    image = plt.imread(str(path))
    #    image.resize((image_size, image_size, 3))
    #    dataset[i] = image
    f = open(out_path, 'w')
    f.write('dummy line\n')
    m = 0
    nnz = 0
    rows = list()
    cols = list()
    for d in sorted(images_path.iterdir()):
        if not d.is_dir() or len(list(d.iterdir())) <= 1:
            continue
        _faces = list(d.iterdir())
        faces = len(_faces)
        for i in range(faces):
            f.write("%s %d %d\n"%(str(d).split('/')[-1], int(str(_faces[i]).split('.')[0].split('_')[-1]), int(str(_faces[i]).split('.')[0].split('_')[-1])))
            for j in range(faces):
                if i == j:
                    continue
                rows.append(m + i)
                cols.append(m + j)
        m += faces
        nnz += faces ** 2 - faces
    label = csr_matrix((np.ones(nnz), (rows,cols)), shape=(m, m))
    save_npz(out_path.replace('txt', 'npz'), label)
    f.close()
    #data.create_dataset('m', data=m)
    #data.create_dataset('nnz', data=nnz)
    #data.create_dataset('rows', shape=(nnz,), data=rows)
    #data.create_dataset('cols', shape=(nnz,), data=cols)
    #data.close()

make(sys.argv[1], out_path=os.path.join(os.path.dirname(sys.argv[1]), os.path.basename(sys.argv[1])+'.txt'))
#make('/tmp2/yusheng/CASIA-maxpy-clean_mtcnnpy_182_subset/', out_path='/tmp2/facenet/te.hdf5')
