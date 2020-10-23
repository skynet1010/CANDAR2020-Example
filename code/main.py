from argparse import ArgumentParser
import os
import selector

def main():
    parser = ArgumentParser()
    parser.add_argument("-bs", "--batch_size",dest="batch_size", default=128,type=int)
    parser.add_argument("-m", "--model-name",dest="model_name", default="resent-18",type=str)
    parser.add_argument("-d", "--data_dir", dest="data_dir", default=os.path.join("..","data"))
    args = parser.parse_args()

    

if __name__ == "__main__":
    main()
