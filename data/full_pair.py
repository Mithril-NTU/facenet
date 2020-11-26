from pathlib import Path
import h5py
import os, sys
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, save_npz
import numpy as np
import tqdm
np.random.seed(0)

image_size = 160
is_full = sys.argv[2]
sample_ratio = 0.01
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
    pnnz = 0
    prows = list()
    pcols = list()
    #seen_dirs = list()
    plabel = list()
    for d in tqdm.tqdm(sorted(images_path.iterdir())):
        if not d.is_dir() or len(list(d.iterdir())) <= 1:
            continue
        _faces = list(d.iterdir())
        faces = len(_faces)
        #if faces > 0:
        #    seen_dirs.append(d)
        for i in range(faces):
            if is_full != 'full':
                f.write("%s %d %d\n"%(str(d).split('/')[-1], i+1, i+1))
            #else:
            #    for sd in seen_dirs:
            #        _sfaces = list(sd.iterdir())
            #        sfaces = len(_sfaces)
            #        for k in range(sfaces):
            #            if np.random.rand() > sample_ratio:
            #                continue
            #            if d == sd:
            #                if i > k:
            #                    f.write("%s %d %d\n"%(str(d).split('/')[-1], i+1, k+1))
            #            else:
            #                f.write("%s %d %s %d\n"%(str(d).split('/')[-1], i+1, str(sd).split('/')[-1], k+1))
            if m + i > 0:
                rnd = np.random.rand(m + i)
                keep_col_idx = np.arange(m + i, dtype=np.int32)[rnd < sample_ratio]
                if keep_col_idx.shape[0] > 0:
                    pnnz += keep_col_idx.shape[0]
                    tmp = np.ones(keep_col_idx.shape[0])
                    tmp[keep_col_idx < m] = -1
                    plabel.append(tmp)
                    prows.extend(list(np.ones(keep_col_idx.shape[0], dtype=np.int32) * (m + i)))
                    pcols.extend(list(keep_col_idx))
            for j in range(faces):
                if i == j:
                    continue
                rows.append(m + i)
                cols.append(m + j)

        m += faces
        nnz += faces ** 2 - faces
    label = csr_matrix((np.ones(nnz), (rows,cols)), shape=(m, m))
    save_npz(out_path.replace('txt', 'npz'), label)
    print(pnnz)
    pidx = csr_matrix((np.hstack(plabel), (prows,pcols)), shape=(m, m))
    save_npz(out_path.replace('txt', 'p.npz'), pidx)
    f.close()
    #data.create_dataset('m', data=m)
    #data.create_dataset('nnz', data=nnz)
    #data.create_dataset('rows', shape=(nnz,), data=rows)
    #data.create_dataset('cols', shape=(nnz,), data=cols)
    #data.close()

make(sys.argv[1], out_path=os.path.join(os.path.dirname(sys.argv[1]), os.path.basename(sys.argv[1])+'.%s.va.txt'%(sys.argv[2])))
#make('/tmp2/yusheng/CASIA-maxpy-clean_mtcnnpy_182_subset/', out_path='/tmp2/facenet/te.hdf5')
