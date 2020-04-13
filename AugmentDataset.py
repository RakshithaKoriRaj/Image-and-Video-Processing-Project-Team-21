import os
import Helper
import Names
from skimage import io, exposure, transform
import matplotlib.pyplot as plt
import random
import math
import time
import Config

'''
    Performs
        contrast 
        rotation 
        cropping (TODO)
    augmentation on the downsampled dataset
'''


### Config

# Only runs for a few images
TESTING_MODE = False
TEST_MODE_NUM = 12

# Debugging displays some images
SHOW_RANDOM_IMAGES = True

DO_CONTRAST = True
DO_ROTATION = True
DO_CROPPING = True


''' ================ Constants / calculated parameters ====== '''

# Generates list 1/3, 1/2, 1/1, 2/1, 3/1, etc.
def generate_gamma_set():
    sideCount = math.ceil(Config.NUM_GAMMA_AUGMENT / 2)

    lst = [i * float(Config.MAX_GAMMA_MULT - 1) / float(sideCount - 1) + 1 for i in range(0, sideCount)]
    lst = lst[:0:-1] + lst
    expo = [-1] * (sideCount - 1) + [1] * sideCount
    return [i ** e for i, e in zip(lst, expo)]

def generate_rotation_set():
    sideCount = math.floor(Config.NUM_ROTATION_AUGMENT / 2)
    lst = range(-sideCount, sideCount + 1)
    return [i / sideCount * Config.MAX_ANGLE for i in lst]

def generate_crop_set():
    def get_lst(maxVal):
        def scale(i):
            return int(math.floor(float(maxVal) * i / (Config.NUM_CROP_AUGMENT - 1)))
        return [scale(i) for i in range(0, Config.NUM_CROP_AUGMENT)]
    return zip(get_lst(CROP_RANGE[0]), get_lst(CROP_RANGE[1]))


GAMMA_RANGE = (1.0 / Config.MAX_GAMMA_MULT, Config.MAX_GAMMA_MULT)
GAMMA_SET = generate_gamma_set()

ROTATION_RANGE = (-Config.MAX_ANGLE, Config.MAX_ANGLE)
ROTATION_SET = generate_rotation_set()

CROP_SIZE = (
    int(Config.DOWNSAMPLE_SIZE[0] * Config.CROP_SCALING),
    int(Config.DOWNSAMPLE_SIZE[1] * Config.CROP_SCALING))
CROP_RANGE = (Config.DOWNSAMPLE_SIZE[0] - CROP_SIZE[0],
              Config.DOWNSAMPLE_SIZE[1] - CROP_SIZE[1])
CROP_SET = generate_crop_set()


''' ========== Paths and folders ============= '''
downsampledDataset = os.path.join(os.getcwd(), Names.basePath, Names.downsampled)
normal = os.path.join(downsampledDataset, Names.normal)
covid = os.path.join(downsampledDataset, Names.covid)

augmentedDataset = os.path.join(os.getcwd(), Names.basePath, Names.augmented)
normalAugmented = os.path.join(augmentedDataset, Names.normal)
covidAugmented = os.path.join(augmentedDataset, Names.covid)

Helper.make_folder(augmentedDataset)
Helper.make_folder(normalAugmented)
Helper.make_folder(covidAugmented)


def augment_dataset(name):
    def load_dataset(name):
        inPath = os.path.join(downsampledDataset, name)
        images = {}

        items = os.listdir(inPath)
        if TESTING_MODE:
            items = items[:TEST_MODE_NUM]
        for item in items:
            fullPath = os.path.join(inPath, item)
            filename, file_extension = os.path.splitext(item)

            images[filename] = io.imread(fullPath, as_grey=True)
        return images

    def show_random_images(images, width, height):
        indices = random.sample(range(0, len(images)), min(len(images), width * height))
        fig, axes = plt.subplots(height, width)

        lst = list(images.values())
        for i, ax in enumerate(axes.flat):
            ax.imshow(lst[indices[i]], cmap='gray', vmin=0, vmax=1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show(block=False)

            #show_image(images[0])

    def print_num(images, step):
        print("{} images in dataset for step {}".format(len(images), step))

    def augment_helper(images, num, doRandom, randomHelper, selectHelper, augmentType, func):
        out = {}
        for name, image in images.items():
            for i in range(0, num):
                if doRandom:
                    result = randomHelper()
                else:
                    result = selectHelper(i)
                out["{}-{}-{}".format(name, augmentType, i)] = func(image, result)
        print_num(out, augmentType)
        if SHOW_RANDOM_IMAGES:
            show_random_images(images, 4, 3)
        return out

    def augment_contrast(images):
        return augment_helper(
            images,
            Config.NUM_GAMMA_AUGMENT,
            Config.DO_GAMMA_RANDOM,
            lambda : random.uniform(GAMMA_RANGE[0], GAMMA_RANGE[1]),
            lambda i : GAMMA_SET[i],
            "contrast",
            lambda image, gamma : exposure.adjust_gamma(image, gamma))

    def augment_rotation(images):
        return augment_helper(
            images,
            Config.NUM_ROTATION_AUGMENT,
            Config.DO_ROTATION_RANDOM,
            lambda : random.uniform(ROTATION_RANGE[0], ROTATION_RANGE[1]),
            lambda i : ROTATION_SET[i],
            "rotation",
            lambda image, rotation : transform.rotate(image, rotation, mode='nearest'))

    def augment_cropping(images):
        def crop(image, corner):
            '''
            print("\nIn shape: {}".format(image.shape))
            print("Corner: {}".format(corner))
            '''
            x1 = corner[0]
            x2 = x1 + CROP_SIZE[0]
            y1 = corner[1]
            y2 = y1 + CROP_SIZE[1]
            #print("{} {} {} {}".format(x1, x2, y1, y2))
            result = image[x1:x2,y1:y2]
            #print("Shape: {}".format(result.shape))
            return result

        return augment_helper(
            images,
            Config.NUM_CROP_AUGMENT,
            Config.DO_CROP_RANDOM,
            lambda : (
                random.randint(0, CROP_RANGE[0]),
                random.randint(0, CROP_RANGE[1])),
            lambda i : CROP_SET[i],
            "cropping",
            lambda image, cropCorner : crop(image, cropCorner))

    def save_images(images, path):
        print("Saving images to {}".format(path))
        for name, image in images.items():
            fullPath = os.path.join(path, "{}.jpg".format(name))
            io.imsave(fullPath, image)

    images = time_function(load_dataset, name, "Load dataset")
    print("{} initial images".format(len(images)))
    show_random_images(images, 4, 3)

    # Contrast augmentation
    if DO_CONTRAST:
        images = time_function(augment_contrast, images, "Augment contrast")

    # Rotation augmentation
    if DO_ROTATION:
        images = time_function(augment_rotation, images, "Augment rotation")

    # Cropping augmentation
    if DO_CROPPING:
        images = time_function(augment_cropping, images, "Augment cropping")

    outPath = os.path.join(augmentedDataset, name)
    #print(outPath)
    save_images(images, outPath)

def time_function(func, input, name):
    print("Timing {}".format(name))
    start = time.time()
    result = func(input)
    end = time.time()
    print("{0} took {1:.2f} seconds".format(name, end - start))
    return result


augment_dataset(Names.normal)
augment_dataset(Names.covid)
plt.show()