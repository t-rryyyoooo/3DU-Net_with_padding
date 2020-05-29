import SimpleITK as sitk
import numpy as np
import argparse
from functions import createParentPath, getImageWithMeta
from pathlib import Path
from extractor import extractor as extor
from tqdm import tqdm
import torch
import cloudpickle
from UNet.model import UNetModel
import re


def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("imageDirectory", help="$HOME/Desktop/data/kits19/case_00000")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("savePath", help="Segmented label file.(.mha)")
    
    parser.add_argument("--patch_size", help="16-48-48", default="16-48-48")
    parser.add_argument("--slide", help="1-1-1")
    parser.add_argument("-g", "--gpuid", help="0 1", nargs="*", default=1, type=int)

    args = parser.parse_args()
    return args

def main(args):
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    """ Slice module. """
    labelFile = Path(args.imageDirectory) / "segmentation.nii.gz"
    imageFile = Path(args.imageDirectory) / "imaging.nii.gz"

    label = sitk.ReadImage(str(labelFile))
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
            )

    extractor.execute()
    image_array_list, l  = extractor.output("Array")

    """ Load model. """

    with open(args.modelweightfile, 'rb') as f:
        model = cloudpickle.load(f)
        model = torch.nn.DataParallel(model, device_ids=args.gpuid)

    model.eval()

    """ Segmentation module. """

    segmented_array_list = []
    cnt = 0
    for image_array in tqdm(image_array_list, desc="Segmenting images...", ncols=60):
        image_array = image_array.transpose(2, 0, 1)
        image_array = torch.from_numpy(image_array[np.newaxis, np.newaxis, ...]).to(device, dtype=torch.float)

        segmented_array = model(image_array)
        segmented_array = segmented_array.to("cpu").detach().numpy().astype(np.float)
        segmented_array = np.squeeze(segmented_array)
        segmented_array = np.argmax(segmented_array, axis=0).astype(np.uint8)
        segmented_array = segmented_array.transpose(1, 2, 0)

        segmented_array_list.append(segmented_array)

    """ Restore module. """
    segmented = extractor.restore(segmented_array_list)

    createParentPath(args.savePath)
    print("Saving image to {}".format(args.savePath))
    sitk.WriteImage(segmented, args.savePath, True)


if __name__ == '__main__':
    args = ParseArgs()
    main(args)
