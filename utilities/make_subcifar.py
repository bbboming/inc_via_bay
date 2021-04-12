from glob import glob
import sys
import os
import argparse
import tqdm
import numpy as np



parser = argparse.ArgumentParser(description='Search Train/Test Folder and Find Match Dataset for Feature Extraction')
parser.add_argument('DIR', help='base_directory')
parser.add_argument('DST', help='dst directory')
args = parser.parse_args()

if __name__ == "__main__":
#    BASESET = ['bear', 'leopard', 'possum', 'raccoon', 'fox', 'tiger', 'wolf', 'camel', 'cattle', 'chimpanzee']
#    INCSET  = ['porcupine',  'lion', 'elephant', 'kangaroo','skunk']
    BASESET = ['bicycle', 'pickup_truck', 'train', 'lawn_mower', 'rocket']
    INCSET  = ['tank', 'bus', 'motorcycle', 'tractor', 'streetcar']
    BASEDIR = args.DIR
    src_train_dir = os.path.join(BASEDIR, "train")
    src_test_dir = os.path.join(BASEDIR, "test")

    train_folder_list = os.listdir(src_train_dir)
    train_folder_list.sort()

    test_folder_list = os.listdir(src_test_dir)
    test_folder_list.sort()
    
    print("Train Folder Size: ", len(train_folder_list))
    print("Test Folder Size: ", len(test_folder_list))
    print(f"BASESET : {BASESET}")
    print(f"INCSET  : {INCSET}")


    #BASESET
    DSTDIR  = os.path.join(BASEDIR, args.DST, "BASE")
    dst_train_dir = os.path.join(DSTDIR, "train")
    dst_test_dir  = os.path.join(DSTDIR, "test")
    if os.path.exists(dst_train_dir):
        print("Remove {:s}".format(dst_train_dir))
        os.system("rm -rf {:s}".format(dst_train_dir))
    if os.path.exists(dst_test_dir):
        print("Remove {:s}".format(dst_test_dir))
        os.system("rm -rf {:s}".format(dst_test_dir))
    os.makedirs(dst_train_dir, exist_ok=True)
    os.makedirs(dst_test_dir, exist_ok=True)
    
    for i, sub in enumerate(tqdm.tqdm(BASESET)):
        folder = sub 
        os.system("cp -rf %s %s"%(os.path.join(src_train_dir, folder), os.path.join(dst_train_dir, folder)))
        os.system("cp -rf %s %s"%(os.path.join(src_test_dir, folder), os.path.join(dst_test_dir, folder)))
        
    #INC SET
    DSTDIR  = os.path.join(BASEDIR, args.DST, "INC1")
    dst_train_dir = os.path.join(DSTDIR, "train")
    dst_test_dir  = os.path.join(DSTDIR, "test")
    if os.path.exists(dst_train_dir):
        print("Remove {:s}".format(dst_train_dir))
        os.system("rm -rf {:s}".format(dst_train_dir))
    if os.path.exists(dst_test_dir):
        print("Remove {:s}".format(dst_test_dir))
        os.system("rm -rf {:s}".format(dst_test_dir))
    os.makedirs(dst_train_dir, exist_ok=True)
    os.makedirs(dst_test_dir, exist_ok=True)

    for i, sub in enumerate(tqdm.tqdm(INCSET)):
        folder = sub 
        os.system("cp -rf %s %s"%(os.path.join(src_train_dir, folder), os.path.join(dst_train_dir, folder)))
        os.system("cp -rf %s %s"%(os.path.join(src_test_dir, folder), os.path.join(dst_test_dir, folder)))
            


    print("Copy Done")
