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

BLACKLIST_FILE_NAMES = [
        "4C4DEFD8-F55D-4588-AAD6-C59017F55966.jpeg",
        "35AF5C3B-D04D-4B4B-92B7-CB1F67D83085.jpeg",
        "44C8E3D6-20DA-42E9-B33B-96FA6D6DE12F.jpeg",
        "254B82FC-817D-4E2F-AB6E-1351341F0E38.jpeg",
        "a1a7d22e66f6570df523e0077c6a5a_jumbo.jpeg",
        "cavitating-pneumonia-4-day0-L.jpg",
        "cavitating-pneumonia-4-day28-L.png",
        "chlamydia-pneumonia-L.png",
        "covid-19-caso-70-1-L.jpg",
        "covid-19-infection-exclusive-gastrointestinal-symptoms-l.png",
        "covid-19-pneumonia-7-L.jpg",
        "covid-19-pneumonia-14-L.png",
        "covid-19-pneumonia-15-L.jpg",
        "covid-19-pneumonia-30-L.jpg",
        "covid-19-pneumonia-evolution-over-a-week-1-day0-L.jpg",
        "D5ACAA93-C779-4E22-ADFA-6A220489F840.jpeg",
        "nejmoa2001191_f1-L.jpeg",
        "nejmoa2001191_f3-L.jpeg",
        "pneumocystis-carinii-pneumonia-1-L.jpg",
        "parapneumonic-effusion-1-L.png",
        "nejmoa2001191_f5-L.jpeg",
        "right-upper-lobe-pneumonia-9-L.jpg",
        ] 

def copy_and_filter_dataset(src, targetCovid):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        #name = item.lower()
        if item not in BLACKLIST_FILE_NAMES and ("covid" in item.lower() or "corona" in item.lower()):
            shutil.copy(s, os.path.join(targetCovid,item))
            

def copy_dataset(src, target):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        shutil.copy(s, os.path.join(target,item))

def rename_all(src):
    count = 0
    for item in os.listdir(src):
        try:
            filename, file_extension = os.path.splitext(os.path.join(src,item))
            os.rename(os.path.join(src,item), os.path.join(src,str(count)) + file_extension)
            count += 1
        except FileExistsError:
            print("Already renamed {}".format(item))

def remove_blacklist(src, target):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        filename, file_extension = os.path.splitext(item)
        if int(filename) in BLACKLIST:
            shutil.move(s, os.path.join(target,item))


dir1 = os.path.join(os.path.abspath(os.getcwd()),"covid-chestxray-dataset-master","images") 
dir2 = os.path.join(os.path.abspath(os.getcwd()),"covid-augmentation-python","dataset")
base = os.path.abspath(os.path.join(os.getcwd(), Names.basePath))
print("base---->"+base)
target = os.path.join(base, Names.merged)
print("target-->"+target)

targetCovid = os.path.join(target, Names.covid)
targetNormal = os.path.join(target, Names.normal)
targetBlacklist = os.path.join(target, Names.blacklist)
print("targetCovid-->"+targetCovid)
print("targetNormal-->"+targetNormal)
print("targetBlacklist-->"+targetBlacklist)
Helper.make_folder(base)
Helper.make_folder(target)
Helper.make_folder(targetCovid)
Helper.make_folder(targetNormal)
Helper.make_folder(targetBlacklist)

copy_and_filter_dataset(dir1, targetCovid)
copy_dataset(os.path.join(dir2,"covid"), targetCovid)
copy_dataset(os.path.join(dir2,"normal"), targetNormal)

rename_all(targetCovid)
rename_all(targetNormal)

#remove_blacklist(targetCovid, targetBlacklist)