from glob import glob
import sys
import os
import argparse
import tqdm



parser = argparse.ArgumentParser(description='Split Dataset folder in two')
parser.add_argument('DIR', help='base_directory')
parser.add_argument('-p', default=0.8, type=float, metavar='Ratio', help='Ration for first divide')
args = parser.parse_args()

if __name__ == "__main__":
    BASEDIR = args.DIR
    train_dir = os.path.join(BASEDIR, "train")
    test_dir = os.path.join(BASEDIR, "test")
    if os.path.exists(train_dir):
        print("Remove {:s}".format(train_dir))
        os.system("rm -rf {:s}".format(train_dir))
 
    if os.path.exists(test_dir):
        print("Remove {:s}".format(test_dir))
        os.system("rm -rf {:s}".format(test_dir))

    folder_list = os.listdir(BASEDIR)
    folder_list.sort()

    os.mkdir(train_dir)
    os.mkdir(test_dir)
    
    total_cnt = len(folder_list)
    
    for folder in tqdm.tqdm(folder_list):
        src = os.path.join(BASEDIR, folder)
        file_list = glob(src+"/*.jpg")
        file_list.sort()

        file_cnt = len(file_list)
        train_cnt = int(float(file_cnt)*args.p)
        test_cnt  = file_cnt - train_cnt

        os.mkdir(os.path.join(train_dir,folder))
        os.mkdir(os.path.join(test_dir, folder))

        for idx, src_file in enumerate(tqdm.tqdm(file_list)):
            dst = os.path.join(train_dir, folder)
            if idx >= train_cnt:
                dst = os.path.join(test_dir, folder)

            os.system("cp -rf  {:s} {:s}/.".format(src_file, dst))
            #os.system("mv  {:s} {:s}/.".format(src_file, dst))

    print("Copy Done")
