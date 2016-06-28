
# coding: utf-8

# # Assess Huntington's disease progression from PET/MR images

import numpy as np
from sklearn.decomposition import PCA

import pandas as pd
import itertools

import glob as glob
import os
import nibabel as nib

from collections import OrderedDict
import re

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import RegularGridInterpolator
from sklearn.utils.extmath import cartesian
import seaborn as sns

def get_region_name(path_to_mask):
    return os.path.basename(path_to_mask).replace('_mask.nii','').split('_')[0]

def load_image(path_to_image, verbose = False):
    """
    Loads an image using nibabel

    Parameters
    ----------
    path_to_image: str with the path to the image
    verbose: boolean whether information about the image is printed out
    """
    img = nib.load(path_to_image)
    if verbose:
        print ("Voxel size for {0}: {1}".format(path_to_mask,
                                                img.header.get_zooms()))
    return img

def get_masked_data(image_data, mask_data):
    """
    Return an image with zeroed voxels outside the mask

    Parameters
    ----------
    image_data: the data for the full image
    mask_data: the data for the mask
    """
    X, Y, Z = (mask_data > 0).nonzero()

    N = len(X)

    data = np.zeros_like(image_data)
    # Replace the original data by the masked one in the original data
    for i in range(0, N):
        data[X[i]][Y[i]][Z[i]] = image_data[X[i]][Y[i]][Z[i]]

    # Return the corrected data
    return data[data > 0]

def extract_voxel_data(subjects, image_filter, skip_regions=[]):

    rows = []

    for s in subjects:
        image = [image for image in s.images
                 if re.match(image_filter, os.path.basename(image))]

        if not image:
            raise Exception('Image not found')
        elif len(image) > 1:
            raise Exception('Found too many images')
        else:
            image = image[0]

        image = load_image(image)
        image_data = image.get_data()

        for mask in s.masks:
            region_name = get_region_name(mask) 
            mask = load_image(mask) 
            mask_data = mask.get_data()
            if region_name not in skip_regions:
                masked_data = get_masked_data(image_data, mask_data)
                rows += [(s.subject_id, region_name, value)
                          for value in masked_data]

    return pd.DataFrame(rows, columns = ['subject_id', 'region', 'value'])

def normalize(df):
    '''
    Normalize a dataframe such as each column has mean = 0 and std = 1.
    '''
    return (df - df.mean()) / df.std()

def extract_features(subjects, features, image_filter, should_normalize = True,
                     groupby=['subject_id', 'region']):
    """
    Extract features from a set of images into a dataframe
    """
    masked_region_df = extract_voxel_data(subjects, image_filter)
    masked_region_features = masked_region_df.groupby(groupby).agg(features).unstack()
    if should_normalize:
        masked_region_features = normalize(masked_region_features)
    return masked_region_df, masked_region_features

class Subject:

    mask_pattern = '_mask'

    def __init__(self, subject_id, subject_dir):
        self.subject_id = subject_id
        self.subject_dir = subject_dir
        all_nifties = glob.glob(os.path.join(subject_dir, '*.nii*'))
        self.masks = find_masks(all_nifties, Subject.mask_pattern)
        self.images = find_images(all_nifties, Subject.mask_pattern)

def find_masks(images, pattern):
    """
    Finds all the masks in a list of images

    Masks are defined as having '_mask' in the name.

    Paramaters
    ----------
    images: a list of string with filenames for images
    """
    return [image for image in images if pattern in image]

def find_images(images, pattern):
    """
    Finds all the images (not masks) in a list of images

    Masks are defined as having '_mask' in the name. An
    image is an image that is not a mask

    Paramaters
    ----------
    images: a list of string with filenames for images
    """
    return [image for image in images if pattern not in image]


def make_subjects(paths, path_to_tracer_data):
    '''
    Create Subject objects from paths
    '''
    subjects = dict(zip(paths, [os.path.join(path_to_tracer_data, path) for path in paths]))
    return [Subject(*item) for item in subjects.items()]

class Slice(object):
    def __init__(self, x, y, z):
        self.cut = None
        self.value = (x, y, z)

class AxialSlice(Slice):
    def __init__(self, z):
        super().__init__(slice(None), slice(None), z)
        self.cut = 'axial'

class SagitalSlice(Slice):
    def __init__(self, x):
        super().__init__(x, slice(None), slice(None))
        self.cut = 'coronal'

class CoronalSlice(Slice):
    def __init__(self, y):
        super().__init__(slice(None), y, slice(None))
        self.cut = 'sagital'

def interpolate_image(image_data, zoom_factor):

    X = np.arange(image_data.shape[0])
    Y = np.arange(image_data.shape[1])

    rgi = RegularGridInterpolator((X, Y), image_data)

    grid_x, grid_y = (np.linspace(0, len(X)-1, zoom_factor*len(X)), 
                      np.linspace(0, len(Y)-1, zoom_factor*len(Y)))

    return rgi(cartesian([grid_x, grid_y])).reshape(grid_x.shape[0], grid_y.shape[0])

def slice_image(data, brain_slice, zoom_factor=3):

    interpolated = interpolate_image(data[brain_slice.value], zoom_factor)
    interpolated = interpolated.transpose()[::-1]

    return interpolated

def heatmap_subplot(img, ax, cmap="gist_heat_r", vmin=0.6, vmax=1.8,
                    cbar=None, cbar_ax=None):

    sns.heatmap(img, cbar=cbar, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, cbar_ax=cbar_ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

def heatmap_grid(images, slices, fig, grid, starting_panel=0, cmap="gist_heat_r",
                 vmin=0.6, vmax=1.8, cbar=None, cbar_ax=None):

    zipped = zip(images * len(slices), slices * len(images))

    for i, (img, sl) in enumerate(zipped):
        ax = plt.Subplot(fig, grid[i + starting_panel])
        heatmap_subplot(slice_image(img, sl), ax, cmap, vmin, vmax, cbar, cbar_ax)
        if isinstance(sl, CoronalSlice):
            ax.invert_xaxis()
        fig.add_subplot(ax)

def despine(axes):
    for ax in axes:
        for sp in ax.spines.values():
            sp.set_visible(False)

def get_median_image(seq_of_images):
    stack = np.stack(seq_of_images)
    return np.median(stack, axis=0)

# Make a color map like the nipy_spectral, but using the white instead of black for the background:

# See http://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps
# sample the colormaps that you want to use. 
colors1 = plt.cm.binary(np.linspace(0., 0.01, 1))
colors2 = plt.cm.nipy_spectral(np.linspace(0.01, 1, 254))

# combine them and build a new colormap
colors = np.vstack((colors1, colors2))
my_spectral = mpl.colors.LinearSegmentedColormap.from_list('my_spectral', colors)
