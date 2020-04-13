
import cv2
import Names
import os
import numpy as np

'''
	Takes all files in the merged dataset folder and
	turns them all into the DOWNSAMPLE_SIZE size
'''

DOWNSAMPLE_SIZE = (128, 128)

mergedDataset = os.path.join(os.getcwd(), Names.basePath, Names.merged)
downsampledDataset = os.path.join(os.getcwd(), Names.basePath, Names.downsampled)

def downsize_dataset(name):
	for item in os.listdir(os.path.join(mergedDataset, name)):
	    fullPath = os.path.join(mergedDataset, name, item)
	    filename, file_extension = os.path.splitext(fullPath)

	    image = np.array(cv2.imread(fullPath))

	    if image.shape[0] < DOWNSAMPLE_SIZE[0] or image.shape[1] < DOWNSAMPLE_SIZE[1]:
	    	print("Warning: Upsampling {}".format(os.path.join(name, item)))
	    	output = cv2.resize(image, DOWNSAMPLE_SIZE, interpolation=cv2.INTER_CUBIC)
	    else:
		    output = cv2.resize(image, DOWNSAMPLE_SIZE, interpolation=cv2.INTER_LINEAR)

	    filename, file_extension = os.path.splitext(item)
	    writePath = os.path.join(downsampledDataset, name, item)
	    #print(writePath)
	    if not cv2.imwrite(writePath, output):
	    	print("Write failed")

downsize_dataset(Names.normal)
downsize_dataset(Names.covid)