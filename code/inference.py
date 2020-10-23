from argparse import Namespace

import torch

from typing import Dict
import time
import os
from utils.dataloader_provider import get_dataloader
from utils.consts import loss_dict, data_compositions, model2data_compositions, model2loss_dict,model2checkpoint_path
from utils.model_manipulator import manipulateModel
from model_exec.test_model import evaluate


def calc_metrics(m):
    m["precision"] = m["true-positive"]/(m["true-positive"]+m["false-positive"])

    return m


def exec_inference(args:Namespace):
    test_data_loader = get_dataloader(args,model2data_compositions[args.model_name],args.model_name)
    
    best_checkpoint = torch.load(os.path.join("..","models",model2checkpoint_path[args.model_name]))

    model = manipulateModel(args.model_name,dim=data_compositions[model2data_compositions[args.model_name]])
    model.load_state_dict(best_checkpoint["model_state_dict"])

    criterion = loss_dict[model2loss_dict[args.model_name]]()

    start = time.time()
    test_metrics = evaluate(model,test_data_loader,criterion)

    test_metrics = calc_metrics(test_metrics)
    test_metrics["exec_time"] = time.time()-start

    print(test_metrics)

    return True

