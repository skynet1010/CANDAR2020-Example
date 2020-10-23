from argparse import ArgumentParser
import os
from inference import exec_inference

def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-name",dest="model_name", default="resnet18",type=str)
    parser.add_argument("-bs", "--batch-size",dest="batch_size", default=128,type=int)
    parser.add_argument("-p", "--parallel-exec",dest="parallel_exec", default=1,type=int)
    args = parser.parse_args()

    exec_inference(args)
    

if __name__ == "__main__":
    main()
