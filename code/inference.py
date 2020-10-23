from argparse import Namespace

import torch

from typing import Dict
import time

from code.utils.dataloader_provider import get_dataloader
from code.utils.consts import optimizer_dict, loss_dict, data_compositions
from code.utils.model_manipulator import manipulateModel
from code.model_exec.test_model import evaluate


def calc_metrics(m):
    m["precision"] = m["true-positive"]/(m["true-positive"]+m["false-positive"])

    return m


def exec_inference(args:Namespace):
    test_data_loader = get_dataloader(args,model2data_compositions[args.model_name],args.model_name)
    
    best_checkpoint = torch.load(best_checkpoint_path)

    model = manipulateModel(args.model_name,dim=data_compositions[model2data_compositions[args.model_name]])
    model.load_state_dict(best_checkpoint["model_state_dict"])

    criterion = loss_dict[model2loss_dict[args.model_name]]

    start = time.time()
    test_metrics = evaluate(model,test_data_loader,criterion)

    test_metrics = calc_metrics(test_metrics)
    test_metrics["exec_time"] = time.time()-start

    return True

