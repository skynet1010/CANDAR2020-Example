from argparse import ArgumentParser
import os
from code.inference import exec_inference

def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-name",dest="model_name", default="resent-18",type=str)
    args = parser.parse_args()

    exec_inference(args)
    

if __name__ == "__main__":
    main()
