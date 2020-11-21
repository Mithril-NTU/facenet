from pathlib import Path
import h5py
import os, sys
import matplotlib.pyplot as plt
clean_self_pair=True
image_size = 160
def make(images_path, out_path):
    images_path = Path(images_path)
    paths = sorted(images_path.glob('*/*'), key=str)
    paths = [p for p in paths if len(os.listdir(os.path.dirname(str(p)))) > 1]
    shape = (len(paths), image_size, image_size, 3)
    data = h5py.File(out_path, 'w')
    dataset = data.create_dataset('images', shape=shape)
    for i, path in enumerate(paths):
        image = plt.imread(str(path))
        image.resize((image_size, image_size, 3))
        dataset[i] = image

    m = 0
    nnz = 0
    rows = list()
    cols = list()
    for d in sorted(images_path.iterdir()):
        if not d.is_dir() or len(list(d.iterdir())) <= 1:
            continue
        faces = len(list(d.iterdir()))
        for i in range(faces):
            for j in range(faces):
                if clean_self_pair and i == j:
                    continue
                rows.append(m + i)
                cols.append(m + j)
        m += faces
        if clean_self_pair:
            nnz += faces ** 2 - faces
        else:
            nnz += faces ** 2
    data.create_dataset('m', data=m)
    data.create_dataset('nnz', data=nnz)
    data.create_dataset('rows', shape=(nnz,), data=rows)
    data.create_dataset('cols', shape=(nnz,), data=cols)
    data.close()

make(sys.argv[1], out_path=os.path.join(os.path.dirname(sys.argv[1]), os.path.basename(sys.argv[1])+'.hdf5'))
#make('/tmp2/yusheng/CASIA-maxpy-clean_mtcnnpy_182_subset/', out_path='/tmp2/facenet/te.hdf5')
