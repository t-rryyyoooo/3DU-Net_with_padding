import sys
import re
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import product
from functions import paddingImage, croppingImage, getImageWithMeta, createParentPath
import re

class extractor():
    def __init__(self, image, label, patch_size=[16, 44, 44], slide=None, padding=None, only_mask=False):
        self.image = image
        self.label = label

        self.patch_size = np.array(patch_size)

        """ Check slide size is correct."""
        if slide is None:
            self.slide = self.patch_size
        else:
            self.slide = np.array(slide)
            if ((self.patch_size % self.slide) != 0).any():
                print("[ERROR] Invalid slide size : {}.".format(self.slide))
                sys.exit()

        self.only_mask = only_mask



    def execute(self):
        """ For restoration. """
        self.meta = {}

        """ Caluculate each padding size for label and image to clip correctly and pad them."""
        padding_size = self.patch_size - (np.array(self.label.GetSize()) % self.patch_size) + (self.patch_size - self.slide)

        lower_padding_size_label = padding_size // 2
        upper_padding_size_label = np.where((padding_size % 2) != 0, padding_size // 2 + 1, padding_size // 2)

        lower_padding_size_image = lower_padding_size_label
        upper_padding_size_image = upper_padding_size_label

        self.meta["lower_padding_size"] = lower_padding_size_label
        self.meta["upper_padding_size"] = upper_padding_size_label
        
        padded_image = paddingImage(self.image, lower_padding_size_image, upper_padding_size_image, mirroring=True)
        padded_label = paddingImage(self.label, lower_padding_size_label, upper_padding_size_label)


        self.meta["padded_label"] = padded_label


        """ Caluculate the patch size. """
        image_patch_size = self.patch_size
        label_patch_size = self.patch_size

        self.meta["patch_size"] = label_patch_size

        """ Clip the patch size image from self.image and self.label. """
        self.image_list = [] 
        self.image_array_list = []
        self.label_list = []
        self.label_array_list = []

        izsize, ixsize, iysize = np.array(padded_image.GetSize()) - image_patch_size
        total_image_patch_idx = [i for i in product(range(0, izsize + 1, self.slide[0]), range(0, ixsize + 1, self.slide[1]), range(0, iysize + 1, self.slide[2]))]

        lzsize, lxsize, lysize = np.array(padded_label.GetSize()) - label_patch_size

        total_label_patch_idx = [i for i in product(range(0, lzsize + 1, self.slide[0]), range(0, lxsize + 1, self.slide[1]), range(0, lysize + 1, self.slide[2]))]

        self.meta["total_patch_idx"] = total_label_patch_idx

        if len(total_image_patch_idx) != len(total_label_patch_idx):
            print("[ERROR] The number of clliped image and label is different.")
            sys.exit()

        with tqdm(total=len(total_image_patch_idx), desc="Clipping image and label...", ncols=60) as pbar:
            for iz, lz in zip(range(0, izsize + 1, self.slide[0]), range(0, lzsize + 1, self.slide[0])):
                for ix, lx in zip(range(0, ixsize + 1, self.slide[1]), range(0, lxsize + 1, self.slide[1])):
                    for iy, ly in zip(range(0, iysize + 1, self.slide[2]), range(0, lysize + 1, self.slide[2])):
                       
                        lz_slice = slice(lz, lz + label_patch_size[0])
                        lx_slice = slice(lx, lx + label_patch_size[1])
                        ly_slice = slice(ly, ly + label_patch_size[2])

                        clipped_label = padded_label[lz_slice, lx_slice, ly_slice]
                        clipped_label_array = sitk.GetArrayFromImage(clipped_label)

                        if self.only_mask and (clipped_label_array == 0).all():
                            pbar.update(1)
                            continue

                        self.label_list.append(clipped_label)
                        self.label_array_list.append(clipped_label_array)

                        iz_slice = slice(iz, iz + image_patch_size[0])
                        ix_slice = slice(ix, ix + image_patch_size[1])
                        iy_slice = slice(iy, iy + image_patch_size[2])
 
                        clipped_image = padded_image[iz_slice, ix_slice, iy_slice]
                        clipped_image_array = sitk.GetArrayFromImage(clipped_image)
                        self.image_list.append(clipped_image)
                        self.image_array_list.append(clipped_image_array)

                        pbar.update(1)


    def output(self, kind = "Array"):
        if kind == "Array":
            return self.image_array_list, self.label_array_list
        elif kind == "Image":
            return self.image_list, self.label_list
        else:
            print("[ERROR] Invalid kind : {}.".format(kind))
            sys.exit()

    def save(self, save_path, patientID):
        save_path = Path(save_path)
        save_image_path = save_path / patientID / "dummy.mha"

        if not save_image_path.parent.exists():
            createParentPath(str(save_image_path))

        with tqdm(total=len(self.image_list), desc="Saving image and label...", ncols=60) as pbar:
            for i, (image, label) in enumerate(zip(self.image_list, self.label_list)):
                save_image_path = save_path / patientID / "image_{:04d}.mha".format(i)
                save_label_path = save_path / patientID / "label_{:04d}.mha".format(i)

                sitk.WriteImage(image, str(save_image_path), True)
                sitk.WriteImage(label, str(save_label_path), True)
                pbar.update(1)



    def restore(self, predict_array_list):
        predict_array = np.zeros_like(sitk.GetArrayFromImage(self.meta["padded_label"]))

        with tqdm(total=len(predict_array_list), desc="Restoring image...", ncols=60) as pbar:
            for pre_array, idx in zip(predict_array_list, self.meta["total_patch_idx"]):
                z_slice = slice(idx[0], idx[0] + self.meta["patch_size"][0])
                x_slice = slice(idx[1], idx[1] + self.meta["patch_size"][1])
                y_slice = slice(idx[2], idx[2] + self.meta["patch_size"][2])


                predict_array[y_slice, x_slice, z_slice] = pre_array
                pbar.update(1)


        predict = getImageWithMeta(predict_array, self.label)
        predict = croppingImage(predict, self.meta["lower_padding_size"], self.meta["upper_padding_size"])
        predict.SetOrigin(self.label.GetOrigin())
        

        return predict






