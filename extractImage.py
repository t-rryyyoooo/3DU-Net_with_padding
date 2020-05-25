import argparse
from pathlib import Path
import SimpleITK as sitk
from extractor import extractor as extor
from functions import getImageWithMeta
import re

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("imageDirectory", help="$HOME/Desktop/data/kits19/case_00000")
    parser.add_argument("saveSlicePath", help="$HOME/Desktop/data/slice/hist_0.0", default=None)
    parser.add_argument("--patch_size", help="28-44-44", default="28-44-44")
    parser.add_argument("--slide", help="2-2-2", default=None)
    parser.add_argument("--only_mask",action="store_true" )

    args = parser.parse_args()
    return args

def main(args):
    labelFile = Path(args.imageDirectory) / 'segmentation.nii.gz'
    imageFile = Path(args.imageDirectory) / 'imaging.nii.gz'

    """ Read image and label. """
    label = sitk.ReadImage(str(labelFile))
    """
    from functions import getImageWithMeta
    import numpy as np
    label = getImageWithMeta(np.ones_like(sitk.GetArrayFromImage(label)), label)
    """
    image = sitk.ReadImage(str(imageFile))

    """ Get the patch size from string."""
    matchobj = re.match("([0-9]+)-([0-9]+)-([0-9]+)", args.patch_size)
    if matchobj is None:
        print("[ERROR] Invalid patch size : {}.".fotmat(args.patch_size))
        sys.exit()

    patch_size = [int(s) for s in matchobj.groups()]

    """ Get the slide size from string."""
    if args.slide is not None:
        matchobj = re.match("([0-9]+)-([0-9]+)-([0-9]+)", args.slide)
        if matchobj is None:
            print("[ERROR] Invalid patch size : {}.".fotmat(args.slide))
            sys.exit()

        slide = [int(s) for s in matchobj.groups()]
    else:
        slide = None


    extractor = extor(
            image = image, 
            label = label,
            patch_size = patch_size, 
            slide = slide, 
            only_mask = args.only_mask
            )

    extractor.execute()
    patientID = args.imageDirectory.split("/")[-1]
    extractor.save(args.saveSlicePath, patientID)
    """
    i, l = extractor.output("Array")
    kk = extractor.restore(l)
    print((sitk.GetArrayFromImage(kk) == 2).all())
    """


if __name__ == '__main__':
    args = ParseArgs()
    main(args)
