import torch
import torchvision
import skimage.transform as transform
import numpy as np
import h5py
import os
import cv2

#TODO: optimize |:->

class Hdf5Dataset(torch.utils.data.Dataset):
    def __init__(self, in_file_name, ds_kind,data_composition, model_key, transform=None):
        super(Hdf5Dataset, self).__init__()
        self.file = h5py.File(in_file_name, "r")
        self.root_ds_dir = "{}/".format(ds_kind)
        self.dir_dict = {"data":"fus_data","labels":"labels"}
        self.n_images, self.nx, self.ny, self.nz = self.file[self.root_ds_dir+self.dir_dict["data"]].shape
        self.transform = transform
        self.model_key = model_key

        DATA_COMPOSITOR_FUNCTOR = {
            "RGB_SLOPE":self.rgb_slope,
            "NDVI_SLOPE":self.ndvi_slope,
            "RGB_NIR_SLOPE":self.rgb_nir_slope
        }
        self.data_composition_function = DATA_COMPOSITOR_FUNCTOR[data_composition]

    def get_sample(self,index):
        return cv2.resize(self.file[self.root_ds_dir+self.dir_dict["data"]][index],(224,224)).transpose(2,0,1)

    def rgb(self,sample):
        return sample[:3,:,:]

    def nir(self,sample):
        return sample[3:4,:,:]

    def slope(self,sample):
        return sample[4:5,:,:]/90.0

    def ndvi(self,sample):
        return sample[6:7,:,:]

    def rgb_slope(self,sample):
        return np.vstack((self.rgb(sample),self.slope(sample)))

    def ndvi_slope(self,sample):
        return np.vstack((self.ndvi(sample),self.slope(sample)))

    def rgb_nir_slope(self,sample):
        return np.vstack((self.rgb(sample),self.nir(sample),self.slope(sample)))


    def __getitem__(self, index):
        sample = self.get_sample(index)
        data={}
        data["imagery"] = self.data_composition_function(sample)
        nr_of_layers = data["imagery"].shape[0] 
        data["labels"] = self.file[self.root_ds_dir+self.dir_dict["labels"]][index]

        return data

    def __len__(self):
        return self.n_images