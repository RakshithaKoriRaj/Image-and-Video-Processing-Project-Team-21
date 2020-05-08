
import cv2
import Names
import os
import numpy as np
import Config
import Helper

'''
	Takes all files in the merged dataset folder and
	turns them all into the Config.DOWNSAMPLE_SIZE size
'''


mergedDataset = os.path.join(os.getcwd(), Names.basePath, Names.merged)
downsampledDataset = os.path.join(os.getcwd(), Names.basePath, Names.downsampled)

Helper.make_folder(downsampledDataset)
Helper.make_folder(os.path.join(downsampledDataset,Names.covid))

Helper.make_folder(os.path.join(downsampledDataset,Names.normal))

def downsize_dataset(name):
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


downsize_dataset(Names.normal)
downsize_dataset(Names.covid)