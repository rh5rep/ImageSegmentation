import os
import glob
import PIL.Image as Image
import torch
from sklearn.model_selection import train_test_split


class DRIVE(torch.utils.data.Dataset):
    def __init__(self, train, transform):
        "Initialization"
        self.transform = transform
        base_path = "/dtu/datasets1/02516/DRIVE/"
        data_path = os.path.join(base_path, "training" if train else "test")
        self.image_paths = sorted(glob.glob(data_path + "/images/*.tif"))
        self.label_paths = sorted(glob.glob(data_path + "/1st_manual/*.gif"))

    def __len__(self):
        "Returns the total number of samples"
        return len(self.image_paths)

    def __getitem__(self, idx):
        "Generates one sample of data"
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)

        if self.transform:
            X = self.transform(image)
            Y = self.transform(label)
        else:
            X = image
            Y = label

        return X, Y


class PH2(torch.utils.data.Dataset):
    def __init__(self, train, transform, test_split=0.2, seed=42):

        self.data_path = os.path.join("/dtu/datasets1/02516/PH2_Dataset_images")

        self.image_paths = sorted(
            glob.glob(
                os.path.join(self.data_path, "**", "*_Dermoscopic_Image/*.bmp"),
                recursive=True,
            )
        )
        self.lesion_paths = sorted(
            glob.glob(
                os.path.join(self.data_path, "**", "*_lesion/*.bmp"), recursive=True
            )
        )
        self.roi_dirs = sorted(
            glob.glob(os.path.join(self.data_path, "**", "*_roi/*.bmp"), recursive=True)
        )

        # Ensure all lists have the same length by filtering out mismatched entries
        min_length = min(
            len(self.image_paths), len(self.lesion_paths), len(self.roi_dirs)
        )
        self.image_paths = self.image_paths[:min_length]
        self.lesion_paths = self.lesion_paths[:min_length]
        self.roi_dirs = self.roi_dirs[:min_length]
        self.transform = transform

        # Debug prints to check lengths
        # print(f'Length of image_paths: {len(self.image_paths)}')
        # print(f'Length of lesion_paths: {len(self.lesion_paths)}')
        # print(f'Length of roi_dirs: {len(self.roi_dirs)}')
        # Split data into train and test sets

        train_paths, test_paths = train_test_split(
            list(zip(self.image_paths, self.lesion_paths, self.roi_dirs)),
            test_size=test_split,
            random_state=seed,
        )

        # Select the appropriate split
        if train:
            self.data = train_paths
        else:
            self.data = test_paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, lesion_path, roi_path = self.data[idx]

        # print(f'{idx = :}')

        image = Image.open(image_path).convert("RGB")
        lesion = Image.open(lesion_path).convert("L")
        roi = Image.open(roi_path).convert("L")

        if self.transform:
            image = self.transform(image)
            lesion = self.transform(lesion)
            # rois = [self.transform(r) for r in roi]
            rois = [self.transform(roi)]
        else:
            rois = [roi]

        return image, lesion, rois
