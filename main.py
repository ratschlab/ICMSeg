import hydra
from omegaconf import DictConfig
import pprint

from utils.options import parse_config
from trainer import ICMSegTrainer
import wandb
import torch
import time
torch.set_float32_matmul_precision('high')
from typing import Any, Dict

def run_model(args: Dict[str, Any]) -> None:
    if not (args is None):
        model = ICMSegTrainer(args)

        if args['run_type'] == "train":
            print(f"Training for {args['num_epochs']} epochs...")
            # print(f"Training for {args['num_steps']} steps...")
            model.train()

        elif args['run_type'] == "test":
            print('Testing...')
            model.test()
        else:
            raise NotImplementedError

    wandb.finish()

    
@hydra.main(version_base=None, config_path="./conf/", config_name="config")
def start_training(cfg: DictConfig) -> None:
    args = parse_config(cfg)
    
    if args['task_name'] == 'debug':
        wandb.init(mode="disabled")

    run_model(args)

if __name__ == '__main__':
    ts = time.time()
    start_training()
    run_time = (time.time() - ts) / 60

    print(f'Run time = {run_time:.1f} min')