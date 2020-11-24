from pathlib import Path
import h5py
import tqdm
import os, sys
import matplotlib.pyplot as plt
import tensorflow as tf
clean_self_pair=sys.argv[2]
image_size = 160
if clean_self_pair == 'va':
    print('Will clean self pair!')
def make(images_path, out_path):
    images_path = Path(images_path)
    paths = sorted(images_path.glob('*/*'), key=str)
    paths = [p for p in paths if len(os.listdir(os.path.dirname(str(p)))) > 1]
    shape = (len(paths), image_size, image_size, 3)
    data = h5py.File(out_path, 'w')
    dataset = data.create_dataset('images', shape=shape)
    for i, path in tqdm.tqdm(enumerate(paths)):
        #image = plt.imread(str(path))
        #image.resize((image_size, image_size, 3))
        file_contents = tf.io.read_file(str(path))
        image = tf.image.decode_image(file_contents, channels=3)
        image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
        image = tf.image.per_image_standardization(image)
        dataset[i] = image.numpy()

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
                if clean_self_pair == 'va' and i == j:
                    continue
                rows.append(m + i)
                cols.append(m + j)
        m += faces
        if clean_self_pair == 'va':
            nnz += faces ** 2 - faces
        else:
            nnz += faces ** 2
    data.create_dataset('m', data=m)
    data.create_dataset('nnz', data=nnz)
    data.create_dataset('rows', shape=(nnz,), data=rows)
    data.create_dataset('cols', shape=(nnz,), data=cols)
    data.close()

make(sys.argv[1], out_path=os.path.join(os.path.dirname(sys.argv[1]), os.path.basename(sys.argv[1])+'.%s.hdf5'%(clean_self_pair)))
#make('/tmp2/yusheng/CASIA-maxpy-clean_mtcnnpy_182_subset/', out_path='/tmp2/facenet/te.hdf5')
