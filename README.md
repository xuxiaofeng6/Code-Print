##### 1.data_generation_pancreas()
```python
import glob
import os
import nibabel as nib
import numpy as np
import skimage.transform as skTrans
from self_supervised_3d_tasks.data_util.nifti_utils import read_scan_find_bbox

def data_generation_pancreas():
    result_path = "/mnt/mpws2019cl1/Task07_Pancreas/images_resized_128_labeled"
    path_to_data = "/mnt/mpws2019cl1/Task07_Pancreas/imagesTr"
    dim = (128, 128, 128)
    list_files_temp = os.listdir(path_to_data)
    for i, file_name in enumerate(list_files_temp):
        path_to_image = "{}/{}".format(path_to_data, file_name)
        try:
            img = nib.load(path_to_image)
            img = img.get_fdata()
            img, bb = read_scan_find_bbox(img)
            img = skTrans.resize(img, dim, order=1, preserve_range=True)
            result = np.expand_dims(img, axis=3)
            file_name = file_name[:file_name.index('.')] + ".npy"
            np.save("{}/{}".format(result_path, file_name), result)
            perc = (float(i) * 100.0) / len(list_files_temp)
            print(f"{perc:.2f} % done")
        except Exception as e:
            print("Error while loading image {}.".format(path_to_image))
            traceback.print_tb(e.__traceback__)
            continue
```

**2.read_scan_find_bbox()**

```python
def read_scan_find_bbox(image, normalize=True, thresh=0.05):
    st_x, en_x, st_y, en_y, st_z, en_z = 0, 0, 0, 0, 0, 0
    if normalize:
        image = norm(image)
    for x in range(image.shape[0]):
        if np.any(image[x, :, :] > thresh):
            st_x = x
            break
    for x in range(image.shape[0] - 1, -1, -1):
        if np.any(image[x, :, :] > thresh):
            en_x = x
            break
    for y in range(image.shape[1]):
        if np.any(image[:, y, :] > thresh):
            st_y = y
            break
    for y in range(image.shape[1] - 1, -1, -1):
        if np.any(image[:, y, :] > thresh):
            en_y = y
            break
    for z in range(image.shape[2]):
        if np.any(image[:, :, z] > thresh):
            st_z = z
            break
    for z in range(image.shape[2] - 1, -1, -1):
        if np.any(image[:, :, z] > thresh):
            en_z = z
            break
    image = image[st_x:en_x, st_y:en_y, st_z:en_z]
    nbbox = np.array([st_x, en_x, st_y, en_y, st_z, en_z]).astype(int)
    return image, nbbox
```

**3.norm()**

```python
def norm(im):
    im = im.astype(np.float32)
    min_v = np.min(im)
    max_v = np.max(im)
    im = (im - min_v) / (max_v - min_v)
    return im
```

