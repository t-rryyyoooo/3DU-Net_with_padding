from UNet.system import UNetSystem
from UNet.modelCheckpoint import BestAndLatestModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
import json
import argparse
import torch

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_json", help="json file for input detail.")

    args = parser.parse_args()

    return args

def main(args):
    torch.manual_seed(0)

    with open(args.input_json, "r") as f:
        input_json = json.load(f)

    dataset_path = input_json["dataset_path"]
    criteria = input_json["criteria"]
    in_channel = input_json["in_channel"]
    num_class = input_json["num_class"]
    epoch = input_json["epoch"]
    batch_size = input_json["batch_size"]
    num_workers = input_json["num_workers"]
    model_savepath = input_json["model_savepath"]
    learning_rate = input_json["learning_rate"]
    gpu_ids = input_json["gpu_ids"]

    api_key = input_json["api_key"]
    project_name = input_json["project_name"]
    experiment_name = input_json["experiment_name"]
    log = input_json["log"]
    comet_logger = CometLogger(
            api_key = api_key,
            project_name = project_name,  
            experiment_name = experiment_name,
            save_dir = log
    )
 
    #torch.manual_seed(0)

    system = UNetSystem(
            dataset_path = dataset_path,
            criteria = criteria,
            in_channel = in_channel,
            num_class = num_class,
            learning_rate = learning_rate,
            batch_size = batch_size,
            num_workers = num_workers, 
            checkpoint = BestAndLatestModelCheckpoint(model_savepath), 
            )

    trainer = pl.Trainer(
            num_sanity_val_steps = 0, 
            max_epochs = epoch,
            checkpoint_callback = None, 
            logger = comet_logger,
            gpus = gpu_ids
        )
    trainer.fit(system)

if __name__ == "__main__":
    args = parseArgs()
    main(args)
