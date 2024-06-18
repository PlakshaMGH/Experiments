import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import utils
from PIL import Image


class BinaryInstrumentDataset(Dataset):
    """
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """

    def __init__(
        self,
        im_root,
        gt_root,
        max_jump,
        num_frames=3,
    ):

        self.im_root = im_root  # Root directory for Images
        self.gt_root = gt_root  # Root directory for ground truth data
        self.max_jump = max_jump  # Maximum distance between frames
        self.num_frames = num_frames  # Number of frames to be sampled
        self.max_num_obj = 1

        # Initialize lists for storing video and frame information
        self.videos = []  # List of videos
        self.frames = {}  # Dictionary mapping video to its frames

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            # List frames in each video directory
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < num_frames:
                continue

            self.frames[vid] = frames
            self.videos.append(vid)

        print(
            f"{len(self.videos)} out of {len(vid_list)} videos accepted in {im_root}."
        )

        # transform to resize and normalise the image
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(size=(384, 384)),
                transforms.ToTensor(),
            ]
        )

        self.normalise = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info["name"] = video

        vid_im_path = os.path.join(self.im_root, video)
        vid_gt_path = os.path.join(self.gt_root, video)
        frames = self.frames[video]

        images = []
        masks = []
        for _ in range(5):
            info["frames"] = []  # Appended with actual frames

            num_frames = self.num_frames
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            # iterative sampling
            frames_idx = [np.random.randint(length)]
            acceptable_set = set(
                range(
                    max(0, frames_idx[-1] - this_max_jump),
                    min(length, frames_idx[-1] + this_max_jump + 1),
                )
            ).difference(set(frames_idx))
            while len(frames_idx) < num_frames:
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(
                    range(
                        max(0, frames_idx[-1] - this_max_jump),
                        min(length, frames_idx[-1] + this_max_jump + 1),
                    )
                )
                acceptable_set = acceptable_set.union(new_set).difference(
                    set(frames_idx)
                )

            frames_idx = sorted(frames_idx)
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_objects = []
            for f_idx in frames_idx:
                jpg_name = frames[f_idx]
                png_name = frames[f_idx]
                info["frames"].append(jpg_name)

                utils.reseed(sequence_seed)
                this_im = Image.open(os.path.join(vid_im_path, jpg_name)).convert("RGB")
                this_im = self.preprocess(this_im)
                this_im = self.normalise(this_im)

                this_gt = Image.open(os.path.join(vid_gt_path, png_name)).convert("P")
                this_gt = self.preprocess(this_gt)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)
            masks = np.stack(masks, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels != 0]

        target_objects = []

        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(
                target_objects, size=self.max_num_obj, replace=False
            )

        info["num_objects"] = max(1, len(target_objects))
        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.num_frames, 384, 384), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, 384, 384), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask = masks == l
            cls_gt[this_mask] = i + 1
            first_frame_gt[0, i] = this_mask[0]
        cls_gt = np.expand_dims(cls_gt, 1)

        # 1 if object exist, 0 otherwise
        selector = [
            1 if i < info["num_objects"] else 0 for i in range(self.max_num_obj)
        ]

        selector = torch.FloatTensor(selector)

        data = {
            "rgb": images,
            "first_frame_gt": first_frame_gt,
            "cls_gt": cls_gt,
            "selector": selector,
            "info": info,
        }

        return data

    def __len__(self):
        return len(self.videos)
