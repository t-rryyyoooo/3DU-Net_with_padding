import shutil
import os
from pathlib import Path

def main():
    org_img_path = Path("/Users/tanimotoryou/Documents/lab/imageData/Abdomen__/RawData/Training/img")
    org_lab_path = Path("/Users/tanimotoryou/Documents/lab/imageData/Abdomen__/RawData/Training/label")
    save_path = Path("/Users/tanimotoryou/Documents/lab/imageData/Abdomen")

    img_glob = sorted(org_img_path.glob("*"))
    lab_glob = sorted(org_lab_path.glob("*"))

    for i, (img, lab) in enumerate(zip(img_glob, lab_glob)):
        save_case_path = save_path / ("case_" + str(i).zfill(2))
        os.makedirs(str(save_case_path), exist_ok=True)
        save_case_img_path = save_case_path / "imaging.nii.gz"
        save_case_lab_path = save_case_path / "segmentation.nii.gz"
        shutil.move(str(img), save_case_img_path)
        shutil.move(str(lab), save_case_lab_path)
    


if __name__ == "__main__":
    main()




