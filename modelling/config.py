import argparse
import warnings

warnings.filterwarnings("ignore")
args = argparse.ArgumentParser()
args.add_argument('--input_dir',
                  type=str,
                  default='/Users/yuewen/Desktop/kaggle-predict-future-sales/data/',
                  help="The directory of the input data.")
args.add_argument('--output_dir',
                  type=str,
                  default='/Users/yuewen/Desktop/kaggle-predict-future-sales/output/',
                  help="the directory to store model's outputs")

args = args.parse_args()


