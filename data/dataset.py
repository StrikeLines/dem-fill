from typing import List
import torch.utils.data as data
from torchvision import transforms
from data.util.dem_transform import DEMNormalize, ToFloat32
import torch.nn.functional as F
from PIL import Image
import os
import torch
import numpy as np
import random
import cv2
import json
from scipy.ndimage import binary_dilation

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images

def pil_loader(path, mode='L'):
    return Image.open(path).convert(mode)

def TIF_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

class DEMDataset(data.Dataset):
    def __init__(
        self,
        data_root: str,
        mask_root: str = None,
        data_len: int = -1,
        data_aug: bool = False,
        image_size: List[int] = [128, 128],
        horizontal_flip: bool = True,
        loader: callable = TIF_loader
    ):
        gt_imgs = make_dataset(data_root)
        mask_imgs = make_dataset(mask_root) if mask_root is not None else None

        if data_len > 0:
            self.gt_imgs = gt_imgs[:int(data_len)]
            self.mask_imgs = mask_imgs[:int(data_len)] if mask_imgs is not None else None
        else:
            self.gt_imgs = gt_imgs
            self.mask_imgs = mask_imgs

        self.loader = loader
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        self.data_aug = data_aug
        self.rotate_angles = [0, 90, 180, 270]
        
        # Load global scaling metadata if available
        self.global_min = None
        self.global_max = None
        try:
            # Check if data_root is a file list, get directory
            if os.path.isfile(data_root):
                tiles_dir = os.path.dirname(data_root)
            else:
                tiles_dir = data_root
                
            global_metadata_path = os.path.join(tiles_dir, "global_scaling.json")
            if os.path.exists(global_metadata_path):
                with open(global_metadata_path, 'r') as f:
                    global_metadata = json.load(f)
                self.global_min = global_metadata.get('global_min')
                self.global_max = global_metadata.get('global_max')
                print(f"Loaded global scaling: min={self.global_min:.3f}, max={self.global_max:.3f}")
                print("Using global scaling to prevent tile edge artifacts")
            else:
                print("No global scaling metadata found - using per-tile scaling (may cause edge artifacts)")
        except Exception as e:
            print(f"Could not load global scaling metadata: {e}")
            print("Falling back to per-tile scaling")

    def __getitem__(self, aug_index):
        ret = {}
        # print("IN __GETITEM__DEMDATSET")

        index = int(aug_index / len(self.rotate_angles)) if self.data_aug else aug_index
        img = self.loader(self.gt_imgs[index])
        img = transforms.ToTensor()(img)
        img = ToFloat32()(img)

        _, h, w = img.shape
        if (h, w) != (128, 128):
            scale_factor = (128 / h, 128 / w)
            img = F.interpolate(img.unsqueeze(0), scale_factor=scale_factor, mode='bicubic', align_corners=False).squeeze(0)
        else:
            scale_factor = (1,1)

        # Augmentation
        if self.data_aug:
            img = F.rotate(img, self.rotate_angles[aug_index % len(self.rotate_angles)])
            if random.random() > 0.5:
                img = F.hflip(img)

        # Load and process mask
        if self.mask_imgs is None:
            y, x, ch, cw = self.get_crop_bbox(img)
            mask = torch.zeros(1, img.shape[1], img.shape[2], dtype=torch.float32)
            mask[:, y:y+ch, x:x+cw] = 1
            mask_scale_factor = (1,1)
        else:
            mask = transforms.ToTensor()(pil_loader(self.mask_imgs[index]))
            mask = ToFloat32()(mask)
            _, mh, mw = mask.shape
            if (mh, mw) != (128, 128):
                mask_scale_factor = (128 / mh, 128 / mw)
                mask = F.interpolate(mask.unsqueeze(0), scale_factor=mask_scale_factor, mode='bicubic', align_corners=False).squeeze(0)
            else:
                mask_scale_factor = (1,1)

            mask[mask > 0] = 1
            mask_np = mask.squeeze().numpy().astype(bool)
            dilated_mask = binary_dilation(mask_np, iterations=2)
            mask = torch.from_numpy(dilated_mask.astype(np.float32)).unsqueeze(0)

        # Use global scaling if available, otherwise fall back to per-tile scaling
        if self.global_min is not None and self.global_max is not None:
            # Use global scaling for consistent tile processing
            min_val = torch.tensor(self.global_min, dtype=img.dtype)
            max_val = torch.tensor(self.global_max, dtype=img.dtype)
            print(f"Using global scaling: {float(min_val):.3f} to {float(max_val):.3f}")
        else:
            # Fall back to per-tile scaling (original behavior)
            valid_mask = mask == 0
            if valid_mask.sum() == 0:
                print(f"Warning - No valid data found after masking in {self.gt_imgs[index]}")
                min_val, max_val = img.min(), img.max()
            else:
                valid_vals = img[valid_mask]
                min_val, max_val = valid_vals.min(), valid_vals.max()

        diff = max_val - min_val
        if diff == 0:
            gt_img = torch.zeros_like(img)
        else:
            gt_img = (img - min_val) / diff
            gt_img = 2 * gt_img - 1
            gt_img = torch.clamp(gt_img, -1, 1)

        cond_img = gt_img.clone()
        cond_img[mask > 0] = -1

        ret['gt_image'] = gt_img
        ret['cond_image'] = cond_img
        ret['mask'] = mask
        ret['path'] = self.gt_imgs[index].rsplit("/")[-1].rsplit("\\")[-1]
        ret['min_val'] = min_val
        ret['max_val'] = max_val
        # print("DATSET - scale_factor",scale_factor)
        ret['scaleFactor'] = scale_factor
        # print(ret)
        return ret

    def __len__(self):
        k = len(self.rotate_angles) if self.data_aug else 1
        return len(self.gt_imgs) * k

    def get_crop_bbox(self, img):
        h, w = img.shape[1], img.shape[2]
        bbox_width = np.random.randint(32, 81)
        bbox_height = np.random.randint(32, 81)
        x_max = w - bbox_width
        y_max = h - bbox_height
        x = np.random.randint(0, x_max + 1)
        y = np.random.randint(0, y_max + 1)
        return y, x, bbox_height, bbox_width
