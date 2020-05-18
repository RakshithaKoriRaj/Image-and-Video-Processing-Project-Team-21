
import cv2
import Names
import os
import numpy as np
import Config
import Helper
import shutil
import random

'''
	Takes all files in the merged dataset folder and
	turns them all into the Config.DOWNSAMPLE_SIZE size
'''





def downsize_dataset(name):
    mergedDataset = os.path.join(os.getcwd(), Names.basePath, Names.merged)
    downsampledDataset = os.path.join(os.getcwd(), Names.basePath, Names.downsampled)
    for item in os.listdir(os.path.join(mergedDataset, name)):
	    fullPath = os.path.join(mergedDataset, name, item)
	    filename, file_extension = os.path.splitext(fullPath)

	    image = np.array(cv2.imread(fullPath))

	    if image.shape[0] < Config.DOWNSAMPLE_SIZE[0] or image.shape[1] < Config.DOWNSAMPLE_SIZE[1]:
	    	print("Warning: Upsampling {}".format(os.path.join(name, item)))
	    	output = cv2.resize(image, Config.DOWNSAMPLE_SIZE, interpolation=cv2.INTER_CUBIC)
	    else:
		    output = cv2.resize(image, Config.DOWNSAMPLE_SIZE, interpolation=cv2.INTER_LINEAR)

	    filename, file_extension = os.path.splitext(item)
	    writePath = os.path.join(downsampledDataset, name, item)
	    if not cv2.imwrite(writePath, output):
	    	print("Write failed")




def seprate_training_test(name,test,train):
    mergedDataset = os.path.join(os.getcwd(), Names.basePath, Names.merged)
    downsampledDataset = os.path.join(os.getcwd(), Names.basePath, Names.downsampled)
    print(name)
    items = os.listdir(os.path.join(mergedDataset, name))
    random.shuffle(items)
    val_size = int(len(items)*Config.VAL_PCT)
    print("Val size:",val_size)
    items_train = items[:-val_size]
    items_test = items[-val_size:]
    for item in items_train:
        fullPath = os.path.join(downsampledDataset, name, item)
        shutil.copy(fullPath, os.path.join(train,item))
    for item in items_test:
        fullPath = os.path.join(downsampledDataset, name, item)
        shutil.copy(fullPath, os.path.join(test,item))


if __name__ == "__main__":
    
    mergedDataset = os.path.join(os.getcwd(), Names.basePath, Names.merged)
    downsampledDataset = os.path.join(os.getcwd(), Names.basePath, Names.downsampled)

    Helper.make_folder(downsampledDataset)
    Helper.make_folder(os.path.join(downsampledDataset,Names.covid))

    Helper.make_folder(os.path.join(downsampledDataset,Names.normal))

    Helper.make_folder(os.path.join(downsampledDataset,Names.covid_test))

    Helper.make_folder(os.path.join(downsampledDataset,Names.normal_test))

    Helper.make_folder(os.path.join(downsampledDataset,Names.covid_train))

    Helper.make_folder(os.path.join(downsampledDataset,Names.normal_train))
    
    
    downsize_dataset(Names.normal)
    downsize_dataset(Names.covid)
    
    seprate_training_test(Names.normal,
                 os.path.join(downsampledDataset,Names.covid_test),
                 os.path.join(downsampledDataset,Names.covid_train))
    seprate_training_test(Names.covid, 
                 os.path.join(downsampledDataset,Names.normal_test),
                 os.path.join(downsampledDataset,Names.normal_train))