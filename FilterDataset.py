'''
    This script takes files from the two datasets and
    puts the relavant files (covid-19 or normal) into the
    merged-dataset folder
'''

import os
import shutil
import Names
import Helper

BLACKLIST = [
    83,  # Side profile
    86,  # ...
    89,  # ...
    91,  # ...
    102, # ...
    111, # ...
    113, # ...
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
base = os.path.abspath(os.path.join(os.getcwd(), Names.basePath))
target = os.path.join(base, Names.merged)
print(target)

targetCovid = os.path.join(target, Names.covid)
targetNormal = os.path.join(target, Names.normal)
targetBlacklist = os.path.join(target, Names.blacklist)

Helper.make_folder(base)
Helper.make_folder(target)
Helper.make_folder(targetCovid)
Helper.make_folder(targetNormal)
Helper.make_folder(targetBlacklist)

copy_and_filter_dataset(dir1, targetCovid)
copy_dataset(dir2 + "\\covid", targetCovid)
copy_dataset(dir2 + "\\normal", targetNormal)

rename_all(targetCovid)
rename_all(targetNormal)

remove_blacklist(targetCovid, targetBlacklist)