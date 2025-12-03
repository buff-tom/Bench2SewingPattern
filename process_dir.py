import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

def find_dirs(root_dir):
    dir_list = glob.glob(os.path.join(root_dir, "*"))
    for dir in dir_list:
        if os.path.isdir(dir):
            print(f"Found directory: {dir}")
            base_name = os.path.basename(dir)
            first_part = "/".join(dir.split('/')[:-1])
            print(f"First part: {first_part}")
            second_part = base_name.split('/')[-1]
            print(f"Second part: {second_part}")
            renamed_dir = second_part.split('_')[0]
            renamed = first_part + "/" + renamed_dir
            print(f"Renamed to: {renamed}")
            os.rename(dir, renamed)   


    

def main():
    root_dir = "/home/user/buff-tomma/Pattern_Making/female_garment/female_asia_front_and_back_garment_with_model"
    find_dirs(root_dir)
    
if __name__ == "__main__":
    main()