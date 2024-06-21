import random
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk
import math
import glob
import os
import h5py
import pydicom
import cv2
from skimage.measure import label
# import gdcm

label_set = ["良性","恶性"]

def label_to_num(label):
    return label_set.index(label)
def num_to_label(n):
    return label_set[n]

import random

class MedicalImageDeal(object):
    def __init__(self, img, path, percent=1):
        self.img = img
        self.percent = percent
        self.p = path

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return ((self.img - self.img.min()) / (self.img.max() - self.img.min())).astype(np.float32)

    @property
    def width_center1(self):
        from skimage import exposure
        image = pydicom.read_file(self.p).pixel_array
        cdf = exposure.cumulative_distribution(image)
        cdf_min = np.min(cdf[1])
        a = np.diff(cdf)
        aa = a[0][500 - cdf_min:1000]
        aa_max = np.max(aa)
        xx = []
        for i in range(aa.shape[0]):
            if aa[i] > aa_max * 0.005:
                xx.append(i + 500)
        ds = np.clip(image, xx[0], xx[len(xx) - 3])
        max = np.max(ds)
        min = np.min(ds)
        ds = (ds-min)/(max-min)
        return ds

def generate_data(root_data,index,save_dir):
    low_energy_fold = os.listdir(os.path.join(root_data,label_set[index],"T1"))
    high_energy_fold = os.listdir(os.path.join(root_data,label_set[index],"T2"))
    enhance_fold = os.listdir(os.path.join(root_data,label_set[index],"enhance"))
    
    patient_names = low_energy_fold
    images = []
    labs = []
    n = 0
    for i in range(len(patient_names)):  # len(patient_names)
        low_energy_list = glob.glob(os.path.join(root_data,label_set[index],"T1",patient_names[i],"*.dcm"))
        high_energy_list = glob.glob(os.path.join(root_data,label_set[index],"T2",patient_names[i],"*.dcm"))
        enhance_list = glob.glob(os.path.join(root_data,label_set[index],"enhance",patient_names[i],"*.dcm"))
        
        low_energy_list = sorted(low_energy_list)
        high_energy_list = sorted(high_energy_list)
        enhance_list = sorted(enhance_list)
        
        for j in range(len(low_energy_list)):
            item = {"low_energy":low_energy_list[j],
                    "high_energy":high_energy_list[j],
                    "enhance":enhance_list[j]}
            images.append(item)
            labs.append(index)
        r_num = 0
    for image_file,lab in zip(images,labs):
        (flepath, flename) = os.path.split(image_file["low_energy"])
        (rename, suffix) = os.path.splitext(flename)
        uid = rename
        low_energy_image = MedicalImageDeal(pydicom.dcmread(image_file["low_energy"]),
                                    image_file["low_energy"]).width_center1
        high_energy_image = MedicalImageDeal(pydicom.dcmread(image_file["high_energy"]),
                                    image_file["high_energy"]).width_center1
        enhance_image = MedicalImageDeal(pydicom.dcmread(image_file["enhance"]),
                                    image_file["enhance"]).width_center1
        R = random.randint(1, 100)
        if R >= 0 and R <= 70:
            f = h5py.File(save_dir+"/train" + '/{}_{}.h5'.format(uid, r_num), 'w')
        elif R > 70 and R <= 90:
            f = h5py.File(save_dir+"/valid" + '/{}_{}.h5'.format(uid, r_num), 'w')
        else:
            f = h5py.File(save_dir+"/test" + '/{}_{}.h5'.format(uid, r_num), 'w')
        f.create_dataset('LOW_ENERGY', data=low_energy_image, compression="gzip")
        f.create_dataset('HIGH_ENERGY', data=high_energy_image, compression="gzip")
        f.create_dataset('ENHANCE', data=enhance_image, compression="gzip")
        f.create_dataset('label', data=lab)
        f.close()
        r_num += 1
        print("process {} uid = {} label={}".format(r_num, uid,lab))

root_data = r"F:\300多的乳腺MR图像"
save_dir = r"F:\乳腺癌数据代码\MF\h5py"

# 读取恶性数据
generate_data(root_data,0,save_dir)
# 读取良性数据
generate_data(root_data,1,save_dir)

