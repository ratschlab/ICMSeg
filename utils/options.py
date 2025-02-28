from argparse import ArgumentParser
from ast import Tuple
from re import T
from omegaconf import DictConfig, OmegaConf
import pprint
from datetime import datetime

from typing import Any, Dict, Tuple


def get_timestamp():
    dt = datetime.now()
    return f"{dt.month:02d}-{dt.day:02d}-{dt.hour:02d}-{dt.minute:02d}"


def convert(input_str: str) -> str:
    if input_str == "None":
        return None
    else:
        return input_str


def flatten_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    args = {}
    for params in cfg:
        if type(cfg[params]) is dict:
            for k in cfg[params]:
                args[k] = convert(cfg[params][k])
        else:
            args[params] = convert(cfg[params])

    if not ("ckpt_file" in args):
        args["ckpt_file"] = None

    return args


def parse_config(cfg: OmegaConf) -> Dict[str, Any]:
    cfg = OmegaConf.to_container(cfg)

    args = flatten_dict(cfg)
    pprint.pprint(args)

    return args


def make_task_name(params: Dict[str, Any]) -> Tuple[str, str]:
    if params["pretrained"]:
        pretrained = "_pretrained"
    else:
        pretrained = ""

    tune = ""
    if "tune" in params:
        if params["tune"]:
            tune = "_tunefull"

    if params["ckpt_in"] is None:
        ckpt_in = ""
    else:
        ckpt_in = f"_ckpt"

    timestamp = get_timestamp()

    # hyparams = f"t{params['temperature']:.1f}"
    hyparams = f"{params['training_domains']}"

    if "version" in params:
        task_name = f"{params['task_name']}_{params['version']}"
    else:
        task_name = params["task_name"]

    if "sublist" in params:
        split = f"/{params['sublist'].split('/')[-1].split('.')[0]}/"
    else:
        split = ""

    meta_folder = (
        f"{params['dataset']}_"
        f"{params['method']}{pretrained}{ckpt_in}{tune}{split}_{task_name}/{hyparams}"
    )

    return meta_folder, f"{meta_folder}/{timestamp}"
