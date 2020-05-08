
''' -------------------------------
DownsampleDataset
---------------------------------'''
DOWNSAMPLE_SIZE = (256, 256)


''' -------------------------------
AugmentDataset
---------------------------------'''

# True: Randomize gamma change within range
# False: Uses equally divided gamma changes
DO_GAMMA_RANDOM = True
# True: Randomize rotation within range
# False: Uses equally divided rotation changes 
DO_ROTATION_RANDOM = True
# True: Randomize cropping area
# False: Uses equally portion cropping 
DO_CROP_RANDOM = True

# Multiplier for each augmentation step
# Should be odd if not random so we maintain
# the 0 angle and 1 gamma (maybe?)
NUM_GAMMA_AUGMENT = 5
NUM_ROTATION_AUGMENT = 5
# Cropping can be whatever multiplier
NUM_CROP_AUGMENT = 5

# Gamma multiplier between 1/MAX_GAMMA_MULT and
# MAX_GAMMA MULT. ex (1/4, 1/2, 1, 2, 4)
MAX_GAMMA_MULT = 4
# Angle changes ex. (-15, -7.5, 0, 7.5, 15)
MAX_ANGLE = 15

# 3/4 size of downsampled images
CROP_SCALING = 3.0/4.0