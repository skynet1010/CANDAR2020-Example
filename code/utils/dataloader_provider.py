import torch
import os
import shutil
import multiprocessing
from code.utils.hdf5_reader import Hdf5Dataset as Dataset
from code.utils.consts import time_stamp
from torch.utils.data.sampler import SubsetRandomSampler
from numpy import floor


def get_dataloader(args,ss, data_composition_key,model_key,validation=True):
    valid_ss = {"resnet18":"ss8","wide_resnet50_2":"ss16","alexnet":"ss32"}

    input_filename = f"train_test_data_{valid_ss[model_key]}_supervised_final.hdf5"

    data_path = os.path.join("..","data")
    
    full_input_filename = os.path.join(data_path,input_filename)
            
    if not os.path.isfile(full_input_filename):
        print(f"Input file {full_input_filename} does not exist! check files!")
        exit(1)

    test_ds = Dataset(full_input_filename,"test",data_composition_key, model_key)
    cpu_count = multiprocessing.cpu_count()

    test_data_loader = torch.utils.data.DataLoader(test_ds,num_workers=cpu_count,batch_size=args.batch_size,pin_memory=True,shuffle=False)

    return test_data_loader