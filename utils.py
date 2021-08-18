import os
import sys
import logging
from pathlib import Path
import yaml

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

# in order to avoid complaining warning from tensorflow logger
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

class MyFilter(logging.Filter):

    def __init__(self, rank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True

class MyLogger(logging.Logger):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addFilter(MyFilter(0))
        self.formatter = logging.Formatter('%(asctime)s %(rank)s [%(levelname)-5s] %(message)s')

    def set_rank(self, rank):
        self.removeFilter(self.filter)
        self.addFilter(MyFilter(rank))

    def set_log_to_stream(self, level=logging.DEBUG):
        chdr = logging.StreamHandler(sys.stdout)
        chdr.setLevel(level)
        chdr.setFormatter(self.formatter)
        self.addHandler(chdr)

    def set_log_to_file(self, log_file, level=logging.DEBUG):
        log_path = Path(log_file).resolve()
        Path.mkdir(log_path.parent, parents=True, exist_ok=True)
        fhdr = logging.FileHandler(log_path)
        fhdr.setLevel(level)
        fhdr.setFormatter(self.formatter)
        self.addHandler(fhdr)

    def set_log_to_slack(self, credential_file, ch_name, level=logging.INFO):
        try:
            credential_path = Path(credential_file).resolve()
            shdr = SlackClientHandler(credential_path, ch_name)
            shdr.setLevel(level)
            shdr.setFormatter(self.formatter)
            self.addHandler(shdr)
        except:
            raise RuntimeError


logging.setLoggerClass(MyLogger)
logger = logging.getLogger("pytorch-cxr")
logger.setLevel(logging.DEBUG)


def print_versions():
    logger.info(f"pytorch version: {torch.__version__}")
    logger.info(f"torchvision version: {torchvision.__version__}")

def get_devices(cuda=None):
    if cuda is None:
        logger.info(f"use CPUs")
        return [torch.device("cpu")]
    else:
        assert torch.cuda.is_available()
        avail_devices = list(range(torch.cuda.device_count()))
        use_devices = [int(i) for i in cuda.split(",")]
        assert max(use_devices) in avail_devices
        logger.info(f"use cuda on GPU {use_devices}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in use_devices])
        return [torch.device(f"cuda:{k}") for k in use_devices]

