import os
import Helper
import Names
from skimage import io, exposure, transform
import matplotlib.pyplot as plt
import random
import math

### Config

# Only runs for a few images
TESTING_MODE = False

SHOW_RANDOM_IMAGES = True
DO_GAMMA_RANDOM = True
DO_ROTATION_RANDOM = True

# Generates list 1/3, 1/2, 1/1, 2/1, 3/1, etc.
def generate_gamma_set(num, maxMult):
    sideCount = math.ceil(num / 2)

    lst = [i * float(maxMult - 1) / float(sideCount - 1) + 1 for i in range(0, sideCount)]
    lst = lst[:0:-1] + lst
    expo = [-1] * (sideCount - 1) + [1] * sideCount
    return [i ** e for i, e in zip(lst, expo)]

def generate_rotation_set(num, maxMult):
    sideCount = math.floor(num / 2)
    lst = range(-sideCount, sideCount + 1)
    return [i / sideCount * maxMult for i in lst]

# Must be odd (so we have 0 gamma change)
NUM_GAMMA_AUGMENT = 5
MAX_GAMMA_MULT = 4
GAMMA_RANGE = (1.0 / MAX_GAMMA_MULT, MAX_GAMMA_MULT)
GAMMA_SET = generate_gamma_set(NUM_GAMMA_AUGMENT, MAX_GAMMA_MULT)

# Must be odd (so we have rotation 0)
NUM_ROTATION_AUGMENT = 5
MAX_ANGLE = 15
ROTATION_RANGE = (-MAX_ANGLE, MAX_ANGLE)
ROTATION_SET = generate_rotation_set(NUM_ROTATION_AUGMENT, MAX_ANGLE)

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
            items = items[:12]
        for item in items:
            fullPath = os.path.join(inPath, item)
            filename, file_extension = os.path.splitext(item)

            images[filename] = io.imread(fullPath, as_grey=True)
        return images

    def show_random_images(images, width, height):
        if SHOW_RANDOM_IMAGES:
            indices = random.sample(range(0, len(images)), min(len(images), width * height))
            fig, axes = plt.subplots(height, width)

            lst = list(images.values())
            for i, ax in enumerate(axes.flat):
                ax.imshow(lst[indices[i]], cmap='gray', vmin=0, vmax=1)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()

            #show_image(images[0])

    def print_num(images, step):
        print("{} images in dataset for step {}".format(len(images), step))

    def augment_helper(images, num, doRandom, valueRange, valueSet, augmentType, func):
        out = {}
        for name, image in images.items():
            for i in range(0, num):
                if doRandom:
                    result = random.uniform(valueRange[0], valueRange[1])
                else:
                    result = valueSet[i]
                out["{}-{}-{}".format(name, augmentType, i)] = func(image, result)
        print_num(out, augmentType)
        return out

    def augment_contrast(images):
        return augment_helper(
            images,
            NUM_GAMMA_AUGMENT,
            DO_GAMMA_RANDOM,
            GAMMA_RANGE,
            GAMMA_SET,
            "contrast",
            lambda image, gamma : exposure.adjust_gamma(image, gamma))

    def augment_rotation(images):
        return augment_helper(
            images,
            NUM_ROTATION_AUGMENT,
            DO_ROTATION_RANDOM,
            ROTATION_RANGE,
            ROTATION_SET,
            "rotation",
            lambda image, rotation : transform.rotate(image, rotation, mode='nearest'))

    def augment_cropping(images):
        print_num(images, "cropping")
        return images

    def save_images(images, path):
        print("Saving images to {}".format(path))
        for name, image in images.items():
            fullPath = os.path.join(path, "{}.jpg".format(name))
            io.imsave(fullPath, image)

    images = load_dataset(name)
    print("{} initial images".format(len(images)))
    show_random_images(images, 4, 3)
    images = augment_contrast(images)
    show_random_images(images, 4, 3)
    images = augment_rotation(images)
    show_random_images(images, 4, 3)
    '''
    images = augment_cropping(images)   
    show_random_images(images, 2, 2)
    '''

    outPath = os.path.join(augmentedDataset, name)
    #print(outPath)
    save_images(images, outPath)

augment_dataset(Names.normal)
augment_dataset(Names.covid)