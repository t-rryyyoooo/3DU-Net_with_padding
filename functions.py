import numpy as np
import os
import SimpleITK as sitk
import tensorflow as tf
import matplotlib.pyplot as plt

def getImageWithMeta(imageArray, refImage, spacing=None, origin=None, direction=None):
    image = sitk.GetImageFromArray(imageArray)
    if spacing is None:
        spacing = refImage.GetSpacing()
    if origin is None:
        origin = refImage.GetOrigin()
    if direction is None:
        direction = refImage.GetDirection()

    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)

    return image

def croppingImage(image, lower_crop_size, upper_crop_size):
    crop_filter = sitk.CropImageFilter()
    crop_filter.SetLowerBoundaryCropSize(lower_crop_size.tolist())
    crop_filter.SetUpperBoundaryCropSize(upper_crop_size.tolist())
    cropped_image = crop_filter.Execute(image)

    return cropped_image

def paddingImage(image, lower_pad_size, upper_pad_size, mirroring = False):
    pad_filter = sitk.MirrorPadImageFilter() if mirroring else sitk.ConstantPadImageFilter()
    pad_filter.SetPadLowerBound(lower_pad_size.tolist())
    pad_filter.SetPadUpperBound(upper_pad_size.tolist())
    padded_image = pad_filter.Execute(image)

    return padded_image


def DICE(trueLabel, result):
    intersection=np.sum(np.minimum(np.equal(trueLabel,result),trueLabel))
    union = np.count_nonzero(trueLabel)+np.count_nonzero(result)
    dice = 2 * intersection / (union + 10**(-9))
   
    return dice

def createParentPath(filepath):
    head, _ = os.path.split(filepath)
    if len(head) != 0:
        os.makedirs(head, exist_ok = True)

# 3D -> 3D or 2D -> 2D
def resampleSize(image, newSize, is_label = False):
    originalSpacing = image.GetSpacing()
    originalSize = image.GetSize()

    if image.GetNumberOfComponentsPerPixel() == 1:
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(image)
        minval = minmax.GetMinimum()
    else:
        minval = None


    newSpacing = [osp * os / ns for osp, os, ns in zip(originalSpacing, originalSize, newSize)]
    newOrigin = image.GetOrigin()
    newDirection = image.GetDirection()

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(newSize)
    resampler.SetOutputOrigin(newOrigin)
    resampler.SetOutputDirection(newDirection)
    resampler.SetOutputSpacing(newSpacing)
    if minval is not None:
        resampler.SetDefaultPixelValue(minval)
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    resampled = resampler.Execute(image)

    return resampled

def advancedSettings(xlabel, ylabel, fontsize=20):
    #plt.figure(figsize=(10,10))
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    #plt.xticks(left + width/2,left)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()
    
    return 

def caluculateAVG(num):
    if len(num) == 0:
        return -1
    else:
        nsum = 0
        for i in range(len(num)):
            nsum += num[i]

        return nsum / len(num)
