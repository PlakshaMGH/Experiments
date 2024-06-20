import datetime
from XMem.util.logger import TensorboardLogger

raw_config = {
    "benchmark": False,
    "bl_root": "../BL30K",
    "davis_root": "../DAVIS",
    "debug": False,
    "deep_update_prob": 0.2,
    "exp_id": "NULL",
    "gamma": 0.1,
    "hidden_dim": 64,
    "key_dim": 64,
    "load_checkpoint": None,
    "load_network": None,
    "log_image_interval": 100,
    "log_text_interval": 50,
    "no_amp": False,
    "num_workers": 0,
    "s3_batch_size": 2,
    "s3_end_warm": 700,
    "s3_finetune": 0,
    "s3_iterations": 3000,
    "s3_lr": 1e-5,
    "s3_num_frames": 16,
    "s3_num_ref_frames": 3,
    "s3_start_warm": 200,
    "s3_steps": [2400],
    "save_checkpoint_interval": 100,
    "save_network_interval": 50,
    "stages": "3",
    "static_root": "../static",
    "value_dim": 512,
    "weight_decay": 0.05,
    "yv_root": "../YouTube",
}

stage = "s3"

config = {
    "batch_size": raw_config[stage + "_batch_size"],
    "iterations": raw_config[stage + "_iterations"],
    "finetune": raw_config[stage + "_finetune"],
    "steps": raw_config[stage + "_steps"],
    "lr": raw_config[stage + "_lr"],
    "num_ref_frames": raw_config[stage + "_num_ref_frames"],
    "num_frames": raw_config[stage + "_num_frames"],
    "start_warm": raw_config[stage + "_start_warm"],
    "end_warm": raw_config[stage + "_end_warm"],
    "max_skip_value": 20,
}

config["num_workers"] = 4

config["deep_update_prob"] = raw_config["deep_update_prob"]
config["weight_decay"] = raw_config["weight_decay"]
config["gamma"] = raw_config["gamma"]
config["amp"] = not raw_config["no_amp"]

config["log_text_interval"] = raw_config["log_text_interval"]
config["log_image_interval"] = raw_config["log_image_interval"]
config["save_network_interval"] = raw_config["save_network_interval"]
config["save_checkpoint_interval"] = raw_config["save_checkpoint_interval"]

config["debug"] = raw_config["debug"]
config["exp_id"] = "EndoVis17_Binary"


def init_logger():
    long_id = "%s_%s" % (
        datetime.datetime.now().strftime("%b%d_%H.%M.%S"),
        config["exp_id"],
    )

    git_info = "XMem"
    logger = TensorboardLogger(config["exp_id"], long_id, git_info)
    # logger.log_string("hyperparams", str(config))
    return logger, long_id
