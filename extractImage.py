import argparse
from pathlib import Path
import SimpleITK as sitk
from extractor import extractor as extor
from functions import getImageWithMeta
import re

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="$HOME/Desktop/data/kits19/case_00000/imaging.nii.gz")
    parser.add_argument("label_path", help="$HOME/Desktop/data/kits19/case_00000/segmentation.nii.gz")
    parser.add_argument("save_slice_path", help="$HOME/Desktop/data/slice/hist_0.0/case_00000", default=None)
    parser.add_argument("--mask_path", help="$HOME/Desktop/data/kits19/case_00000/label.mha")
    parser.add_argument("--image_patch_size", help="16-48-48", default="16-48-48")
    parser.add_argument("--label_patch_size", help="16-48-48", default="16-48-48")
    parser.add_argument("--slide", help="2-2-2", default=None)
    parser.add_argument("--threshold", help="Criteria not to save", default=0, type=float)

    args = parser.parse_args()
    return args

def main(args):
    """ Read image and label. """
    label = sitk.ReadImage(args.label_path)
    image = sitk.ReadImage(args.image_path)
    if args.mask_path is not None:
        mask = sitk.ReadImage(args.mask_path)
    else:
        mask = None

    """ Get the patch size from string."""
    matchobj = re.match("([0-9]+)-([0-9]+)-([0-9]+)", args.image_patch_size)
    if matchobj is None:
        print("[ERROR] Invalid patch size : {}.".fotmat(args.image_patch_size))
        sys.exit()

    image_patch_size = [int(s) for s in matchobj.groups()]
    
    """ Get the patch size from string."""
    matchobj = re.match("([0-9]+)-([0-9]+)-([0-9]+)", args.label_patch_size)
    if matchobj is None:
        print("[ERROR] Invalid patch size : {}.".fotmat(args.label_patch_size))
        sys.exit()

    label_patch_size = [int(s) for s in matchobj.groups()]


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
            mask = mask,
            image_patch_size = image_patch_size, 
            label_patch_size = label_patch_size, 
            slide = slide, 
            phase="train",
            threshold = args.threshold
            )

    extractor.execute()
    extractor.save(args.save_slice_path)


if __name__ == '__main__':
    args = ParseArgs()
    main(args)
