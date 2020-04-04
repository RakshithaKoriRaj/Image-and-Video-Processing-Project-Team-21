'''
    This script takes files from both datasets and merges them
    together in the merged-dataset folder

    It also takes the files from the covid-chestxray-dataset-master
    dataset and filters out the ones that are acutal covid-19
    patients and not for other diseases (I think)
'''

import os
import shutil

BLACKLIST = [
    83,  # Side profile
    89,  # ...
    91,  # ...
    102, # ..
    111, # ...
    113,
]

def copy_and_filter_dataset(src, target):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        name = item.lower()
        if "covid" in name or "corona" in name:
            shutil.copy(s, target + "\\" + item)

def copy_dataset(src, target):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        shutil.copy(s, target + "\\" + item)

def rename_all(src):
    count = 0
    for item in os.listdir(src):
        try:
            filename, file_extension = os.path.splitext(src + "\\" + item)
            os.rename(src + "\\" + item, src + "\\" + str(count) + file_extension)
            count += 1
        except FileExistsError:
            print("Already renamed {}".format(item))

def remove_blacklist(src, target):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        filename, file_extension = os.path.splitext(item)
        if int(filename) in BLACKLIST:
            shutil.move(s, target + "\\" + item)


dir1 = os.path.abspath(os.getcwd()) + "\\covid-chestxray-dataset-master\\images"
dir2 = os.path.abspath(os.getcwd()) + "\\covid-augmentation-python\\dataset"
target = os.path.abspath(os.getcwd()) + "\\merged-dataset"

targetCovid = target + "\\covid"
targetNormal = target + "\\normal"
targetBlacklist = target + "\\blacklist"

def try_make(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        print(path + " already exists")

try_make(target)
try_make(targetCovid)
try_make(targetNormal)
try_make(targetBlacklist)

copy_and_filter_dataset(dir1, targetCovid)
copy_dataset(dir2 + "\\covid", targetCovid)
copy_dataset(dir2 + "\\normal", targetNormal)

rename_all(targetCovid)
rename_all(targetNormal)

remove_blacklist(targetCovid, targetBlacklist)