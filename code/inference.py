from argparse import Namespace

import torch

from typing import Dict
import time

from code.utils.dataloader_provider import get_dataloader
from code.utils.consts import optimizer_dict, loss_dict, data_compositions
from code.utils.model_manipulator import manipulateModel
from code.model_exec.test_model import evaluate

args = None
ss = None
data_composition_key = None
model_key = None

def calc_metrics(m):
    m["precision"] = m["true-positive"]/(m["true-positive"]+m["false-positive"])

    return m


def inference(parameters:Dict):
    test_data_loader = get_dataloader(args,ss,data_composition_key, model_key)
    
    best_checkpoint = torch.load(best_checkpoint_path)

    model = manipulateModel(model_key,args.is_feature_extraction,data_compositions[data_composition_key])
    model.load_state_dict(best_checkpoint["model_state_dict"])

    criterion = parameters.get("criterion")

    start = time.time()
    test_metrics = evaluate(model,test_data_loader,criterion)

    test_metrics = calc_metrics(test_metrics)
    test_metrics["exec_time"] = time.time()-start

    return True

