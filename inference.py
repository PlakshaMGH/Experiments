import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from skimage import io
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification import Dice
import numpy as np
from XMem.inference.inference_core import InferenceCore
from XMem.inference.interact.interactive_utils import (
    torch_prob_to_numpy_mask,
    index_numpy_to_one_hot_torch,
)
from XMem.model.network import XMem
import gc
from torchmetrics.classification import Dice, BinaryJaccardIndex
import pandas as pd

device = "cuda"

# default configuration
config = {
    "top_k": 30,
    "mem_every": 5,
    "deep_update_every": -1,
    "enable_long_term": True,
    "enable_long_term_count_usage": True,
    "num_prototypes": 128,
    "min_mid_term_frames": 5,
    "max_mid_term_frames": 10,
    "max_long_term_elements": 10000,
}


def getIoU(pred_frames, gt_path):
    metric = BinaryJaccardIndex()
    dice_metric = Dice()

    IoU = []
    dice = []

    for frame_name, mask in pred_frames.items():
        mask = torch_prob_to_numpy_mask(mask)
        try:
            truth_mask = io.imread(gt_path / frame_name)
        except FileNotFoundError:
            continue
        truth_mask = np.where(truth_mask == 255, 1, truth_mask)
        if np.sum(truth_mask) == 0:
            continue
        truth_mask = torch.tensor(truth_mask)
        IoU.append(metric(torch.tensor(mask), truth_mask).item())
        dice.append(dice_metric(torch.tensor(mask), truth_mask).item())

    meanIoU = sum(IoU) / len(IoU)
    meanDice = sum(dice) / len(dice)

    return meanIoU, IoU, meanDice, dice


im_normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


def resize_mask(mask, size):
    mask = mask.unsqueeze(0).unsqueeze(0)
    h, w = mask.shape[-2:]
    min_hw = min(h, w)
    return F.interpolate(
        mask, (int(h / min_hw * size), int(w / min_hw * size)), mode="nearest"
    )[0]


def singleVideoInference(images_paths, first_mask, processor, size=-1):
    predictions = {}
    frames = {}
    with torch.cuda.amp.autocast(enabled=True):

        images_paths = sorted(images_paths)

        # First Frame
        frame = io.imread(images_paths[0])
        shape = frame.shape[:2]
        if size < 0:
            im_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    im_normalization,
                ]
            )
        else:
            im_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    im_normalization,
                    transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
                ]
            )

        frame_torch = im_transform(frame).to(device)
        first_mask = first_mask.astype(np.uint8)
        if size > 0:
            first_mask = torch.tensor(first_mask).to(device)
            first_mask = resize_mask(first_mask, size)
        else:
            NUM_OBJECTS = 1  # Binary Segmentation
            first_mask = index_numpy_to_one_hot_torch(first_mask, NUM_OBJECTS + 1).to(
                device
            )
            first_mask = first_mask[1:]

        prediction = processor.step(frame_torch, first_mask)

        for image_path in tqdm(images_paths[1:]):
            frame = io.imread(image_path)
            # convert numpy array to pytorch tensor format
            frame_torch = im_transform(frame).to(device)

            prediction = processor.step(frame_torch)
            # Upsample to original size if needed
            if size > 0:
                prediction = F.interpolate(
                    prediction.unsqueeze(1), shape, mode="bilinear", align_corners=False
                )[:, 0]
            predictions[image_path.name] = prediction
            frames[image_path.name] = frame

    return frames, predictions


def firstMaskGT(image_files, mask_folder):
    image_files = sorted(image_files)

    for idx, image_path in enumerate(image_files):

        # Getting the Path to Mask Ground Truth using RGB Image path
        mask_path = mask_folder / image_path.parent.name / image_path.name
        print(mask_folder)

        mask = io.imread(mask_path)
        # All 255 Values replaced with 1, other values remain as it is.
        mask = np.where(mask == 255, 1, mask)

        if np.sum(mask) > 0:
            return mask, idx

    return None, -1


def doInference(
    network_path,
    config,
    frames_folder,
    mask_folder,
    subset=None,
    pred_mask_folder=None,
    size=-1,
):
    overallIoU = []
    overallDice = []
    for video_folder in sorted(frames_folder.iterdir()):

        print(video_folder)

        # Clearing GPU Cache
        torch.cuda.empty_cache()
        network = XMem(config, network_path).eval().to(device)
        processor = InferenceCore(network, config=config)
        NUM_OBJECTS = 1  # Binary Segmentation
        processor.set_all_labels(range(1, NUM_OBJECTS + 1))

        # All Images
        image_files = sorted(list(video_folder.iterdir()))
        if pred_mask_folder:
            mask_path = [
                i for i in pred_mask_folder.iterdir() if video_folder.name in i.name
            ][0]
            mask = io.imread(mask_path)
            # All 0 pixel is 0, everything else(which is mask) is 1
            mask = np.where(mask == 0, 0, 1)
            # seq_01_0.png -> Two Splits, one on '_', other on '.'
            start_idx = int((mask_path.name.split("_")[-1]).split(".")[0])
        else:  # Ground Truth
            mask, start_idx = firstMaskGT(image_files, mask_folder)

        print(f"Running Inference on {video_folder.name}...")
        frames, predictions = singleVideoInference(
            image_files[start_idx:], mask, processor, size=size
        )
        IoU, _, dice, _ = getIoU(predictions, mask_folder / video_folder.name)
        print(f'Video "{video_folder.name}", mean IoU is: {IoU}')
        print(f'Video "{video_folder.name}", mean dice is: {dice}')

        overallIoU.append(IoU)
        overallDice.append(dice)
        print()

        del network, processor
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Average IoU over all videos is: {sum(overallIoU)/len(overallIoU)}.")
    print(f"Average Dice over all videos is: {sum(overallDice)/len(overallDice)}.")

    return overallIoU, overallDice


def run_inference_all_models():
    WEIGHTS_FOLDER = Path("./artifacts/saved_models")

    DATA_FOLDER = Path("./data")
    VIDEOS_PATH = DATA_FOLDER / "frames" / "test"
    MASKS_PATH = DATA_FOLDER / "masks" / "test" / "binary_masks"

    network_paths = [pth for pth in WEIGHTS_FOLDER.iterdir() if pth.is_file()]
    network_name = [pth.name.split("_")[4].split(".")[0] for pth in network_paths]
    network_name = [int(pth) for pth in network_name]

    network_scores = pd.DataFrame()

    for idx, network_path in enumerate(network_paths):
        with torch.inference_mode():
            score = doInference(network_path, config, VIDEOS_PATH, MASKS_PATH, size=384)
            network_scores[network_name[idx]] = (np.mean(score[0]), np.mean(score[1]))

    network_scores.to_json("./saved_scores.json")


if __name__ == "__main__":
    run_inference_all_models()
