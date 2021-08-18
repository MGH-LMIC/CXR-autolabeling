import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

from utils import logger, print_versions, get_devices
from model import BaseModel
from data import CxrDataset, EXT_DATA_BASE, label_name

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

target_view = 'pa' # 'pa' or 'ap'
label_v5 = [ f'{target_view}',
       'Bone>Fracture>.', 'Bone>Non-fracture>.', 'Diaphragm>Diaphragm>.',
       'Foreign body>.>.', 'Hilar/mediastinum>Aorta>.',
       'Hilar/mediastinum>Cardiomegaly>.', 'Hilar/mediastinum>Hilar area>.',
       'Hilar/mediastinum>Mediastinum>.',
       'Lung density>Decreased density (Lucency)>Cavity/Cyst',
       'Lung density>Decreased density (Lucency)>Emphysema',
       'Lung density>Increased lung density>Atelectasis',
       'Lung density>Increased lung density>Nodule/mass',
       'Lung density>Increased lung density>Other interstitial opacity',
       'Lung density>Increased lung density>Pulmonary edema',
       'Lung density>Increased lung density>pneumonia',
       'Lung volume>Decreased lung volume>.',
       'Lung volume>Increased lung volume>.', 'Pleura>Other pleural lesions>.',
       'Pleura>Pleural effusion>.', 'Pleura>Pneumothorax>.']

print_label_name =[
        'view',
        'fracture', 'non_fracture', 'diaphragm',
        'foreign_body', 'aorta',
        'cardiomegaly', 'hilar_area',
        'mediastinum',
        'cavity_cyst',
        'emphysema',
        'atelectasis',
        'nodule_mass',
        'other_int_opacity',
        'pulmonary_edema',
        'pneumonia',
        'decreased_lung_volume',
        'increased_lung_volume', 'other_pleural_lesions',
        'pleural_effusion', 'pneumothorax']


def initialize(args, fl_demo=False):
    if fl_demo:
        runtime_path = Path(args.runtime_dir).resolve()
    else:
        runtime_path = Path('./runtime', args.runtime_dir).resolve()


    logger.handlers.clear()
    # set logger
    log_file = f"train.log"
    logger.set_log_to_stream()
    logger.set_log_to_file(runtime_path.joinpath(log_file))

    # print versions after logger.set_log_to_file() to log them into file
    print_versions()
    #logger.info(f"runtime commit: {get_commit()}")
    logger.info(f"runtime path: {runtime_path}")

    # for fixed random indices
    random_seed = 20      #for Pytorch 1.2.0 + DenseNet121 #basic
    logger.info(f"random seed for reproducible: {random_seed}")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # check gpu device
    device = get_devices(args.cuda)

    return runtime_path, device


class BaseEnvironment:
    def __init__(self, device, mtype=0, in_dim=1, out_dim=31, model_file=None, tf_learning=None, name_model=None):
        self.device = device[0]
        self.type = mtype
        self.num_gpu = len(device)
        pretrained = False if tf_learning == None else True

        if mtype == 0:
            self.model = BaseModel(in_dim, out_dim, name_model=name_model, pretrained=pretrained, tf_learning=tf_learning)
        elif mtype == 1:
            self.model = TwoInputBaseModel(in_dim, out_dim)
        elif mtype == 2:
            self.model = VaeModel(nm_ch=in_dim, dm_lat=out_dim)
        elif mtype == 3:
            self.model = ExtdModel(in_dim=in_dim, out_dim=out_dim, name_model=name_model, pretrained=pretrained, tf_learning=tf_learning)
        else:
            raise RuntimeError

        if model_file is not None:
            self.load_model(model_file)

        self.model = nn.DataParallel(self.model, device_ids=device)
        self.model.to(self.device)

    def load_model(self, filename):
        filepath = Path(filename).resolve()
        logger.debug(f"loading the model from {filepath}")
        states = torch.load(filepath, map_location=self.device)
        try:
            self.model.load_state_dict(states, strict=True)
        except:
            model_dict = self.model.state_dict()
            # 1. filter out unnecessary keys
            states = {k: v for k, v in states.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(states)
            # 3. load the new state dict
            self.model.load_state_dict(model_dict)

class TestEnvironment(BaseEnvironment):
    def __init__(self, device, mtype=0, in_dim=1, out_dim=31, name_labels=None, name_paths=None, testset_csv=None, name_model=None, r_seed=-1):
        self.arch = name_model

        label_grp = label_name
        name_labels = label_v5 if name_labels == None else name_labels
        out_dim = len(name_labels)

        super().__init__(device, mtype=mtype, in_dim=in_dim, out_dim=out_dim, name_model=name_model)
        print(f'{name_paths}')
        dataset_filename = f'post2015_mgh_cxr_{target_view}_test_v5_cat3_golden_th02.csv' if testset_csv == None else testset_csv
        self.test_set = CxrDataset(EXT_DATA_BASE, dataset_filename, num_labels=out_dim, name_labels=name_labels, name_paths=name_paths, fl_balance=False, r_seed=r_seed)

        self.setup_dataset()

    def setup_dataset(self):
        pin_memory = True if self.device.type == 'cuda' else False
        if self.arch == 'resnet152':
            batch_size = (self.num_gpu*8)
        elif self.arch == 'resnext101_32x8d':
            batch_size = (self.num_gpu*20)
        elif self.arch == 'densenet161':
            batch_size = (self.num_gpu*8*3)
        else:
            batch_size = (self.num_gpu*12*3)
            #batch_size = (self.num_gpu*12)

        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, num_workers=(self.num_gpu*8), shuffle=False, pin_memory=pin_memory)
        self.gradcam_loader = DataLoader(self.test_set, batch_size=1, num_workers=0, shuffle=False, pin_memory=pin_memory)

        self.labels = self.test_set.labels
        self.out_dim = len(self.labels)

        nm_count = len(self.test_loader.dataset)
        logger.info(f"using ({nm_count}) entries for testing")


