import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

def find_files(root_dir):
    dir_list = glob.glob(os.path.join(root_dir, "*"))
    print(dir_list)
    

def main():
    root_dir = ""
    find_files(root_dir)
    
if __name__ == "__main__":
    main()