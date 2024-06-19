import torch
from pathlib import Path
from dataset import BinaryInstrumentDataset
import configs as configs
import utils
from XMem.model.trainer import XMemTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"Training on {device}.")

MAIN_FOLDER = Path("./data")
TRAIN_VIDEOS_PATH = MAIN_FOLDER / "frames/train"
TRAIN_MASKS_PATH = MAIN_FOLDER / "masks/train/binary_masks"
SAVE_DIR = Path("./artifacts/saved_models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Set seed to ensure the same initialization
utils.reseed(42)

_, long_id = configs.init_logger()

train_dataset = BinaryInstrumentDataset(
    TRAIN_VIDEOS_PATH,
    TRAIN_MASKS_PATH,
    max_jump=configs.config["max_skip_value"],
    num_frames=configs.config["num_frames"],
)

train_loader = DataLoader(
    train_dataset,
    configs.config["batch_size"],
    num_workers=configs.config["num_workers"],
    drop_last=True,
)

model = XMemTrainer(
    configs.config,
    save_path=SAVE_DIR / long_id,
    local_rank=0,
    world_size=1,
).train()
model.load_network("./artifacts/pretrained_weights/XMem.pth")

total_epochs = configs.config["iterations"]
print(f"Training model for {total_epochs} epochs.")
## Train Loop
model.train()

pbar = tqdm(range(total_epochs), unit="Epoch")
for epoch in pbar:
    for data in train_loader:
        data["rgb"] = data["rgb"].cuda()
        data["first_frame_gt"] = data["first_frame_gt"].cuda()
        data["cls_gt"] = data["cls_gt"].cuda()
        data["selector"] = data["selector"].cuda()
        loss = model.do_pass(data, 50 + epoch)
        pbar.set_postfix(loss=loss)

    if epoch % 100 == 0:
        model.save_network(epoch)
