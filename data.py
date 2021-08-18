from pathlib import Path

import bisect
from PIL import Image
from tqdm import tqdm
import pandas as pd
import imageio

import torch
import torchvision.transforms as tfms
from torch.utils.data import Dataset, ConcatDataset, Subset

from utils import logger

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_BASE = Path('data').resolve()
EXT_DATA_BASE = DATA_BASE.joinpath('pa_experiment').resolve()

label_name = ['Bone>Fracture>.', 'Bone>Non-fracture>.', 'Diaphragm>Diaphragm>.',
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

folder_name = ['b_f', 'b_nf', 'd_d', 'fb', 'hm_a',
        'hm_c', 'hm_ha', 'hm_m', 'ld_dd_cc', 'ld_dd_e',
        'ld_ild_a', 'ld_ild_nm', 'ld_ild_oio', 'ld_ild_pe', 'ld_ild_p',
        'lv_dlv', 'lv_ilv', 'p_opl', 'p_pe', 'p_p']


def _tb_load_manifest(file_path, num_labels=31, name_labels=None, name_paths=None, ext_data=False, fl_balance=False, r_seed=-1):
    if not file_path.exists():
        logger.error(f"manifest file {file_path} not found.")
        raise RuntimeError

    logger.debug(f"loading dataset manifest {file_path} ...")
    df = pd.read_csv(str(file_path)).fillna(0)

    if (not ext_data) and (True): # using the clean-set
        # cleanset
        if True:
            ## MGH validation set
            df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)]
            if r_seed != -1:
                df = df.sample(n=1000, replace=True, random_state=r_seed)

            ## CheXpert trainset
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][0:100]

            if (True): # in order to add clinical information to network
                df['ScaledSex'] = df.sex.replace(0, -1)
                weight_gender = 10
                weight_age = 100
                min_age = 11.0
                max_age = 100.0
                df['ScaledAge'] = (df.PatientAge-min_age)/(max_age-min_age)
                df.ScaledAge = weight_age * (df.ScaledAge - 0.5)
                df['ScaledSex'] = weight_gender * df.ScaledSex

            df.reset_index(drop=True, inplace=True)

    else:
        try:
            df['ScaledSex'] = df.sex.replace(0, -1)
            weight_gender = 10
            weight_age = 100
            min_age = 11.0
            max_age = 117.0
            df['ScaledAge'] = (df.PatientAge-min_age)/(max_age-min_age)
            df.ScaledAge = weight_age * (df.ScaledAge - 0.5)
            df['ScaledSex'] = weight_gender * df.ScaledSex
        except:
            df['ScaledAge'] = 0
            df['ScaledSex'] = 0

    LABELS = df.columns[-(num_labels+1):-1] if name_labels == None else name_labels
    labels = df[LABELS].astype(int)
    paths = df['PATH'] if name_paths == None else df[name_paths]
    ages = df['ScaledAge'].astype(float)
    genders = df['ScaledSex'].astype(float)
    df_tmp = pd.concat([paths, ages, genders, labels], axis=1)

    entries = df_tmp

    logger.debug(f"{len(entries)} entries are loaded.")
    return entries

# data augmentation - 512
train_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.Resize(562, Image.LANCZOS),
    tfms.RandomRotation((-10, 10)),
    tfms.RandomCrop((512, 512)),
    tfms.RandomHorizontalFlip(p=0.01), #with 1% horizontal flip
    tfms.ToTensor(),
])

test_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.Resize((512, 512), Image.LANCZOS),
    tfms.ToTensor(),
])

def get_image(img_path, transforms):
    image = imageio.imread(img_path)
    image_tensor = transforms(image)
    image_tensor = image_tensor[:1, :, :]
    return image_tensor


class CxrDataset(Dataset):
    transforms = train_transforms

    def __init__(self, base_path, manifest_file, num_labels=31, name_labels=None, name_paths=None, ext_data=False, csv_path=None, fl_balance=False, r_seed=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest_path = base_path.joinpath(manifest_file).resolve() if csv_path == None else csv_path.joinpath(manifest_file).resolve()
        self.entries = _tb_load_manifest(manifest_path, num_labels=num_labels, name_labels=name_labels, name_paths=name_paths, ext_data=ext_data, fl_balance=fl_balance, r_seed = r_seed)
        self.base_path = base_path
        self.name_labels = name_labels

    def __getitem__(self, index):
        # need to debug
        def get_entries(index):
            df = self.entries.loc[index]
            paths = self.base_path.joinpath(df[0]).resolve()
            label = df[3:].tolist() if self.name_labels == None else df[self.name_labels].tolist()
            age = df[1]
            gender = df[2]

            return paths, label, age, gender

        img_path, label, age, gender = get_entries(index)
        image_tensor = get_image(img_path, CxrDataset.transforms)
        target_tensor = torch.FloatTensor(label)
        clinic_tensor = torch.FloatTensor([age, gender])

        return image_tensor, target_tensor, clinic_tensor


    def __len__(self):
        return len(self.entries)

    def get_label_counts(self, indices=None):
        df = self.entries if indices is None else self.entries.loc[indices]
        counts = [df[x].value_counts() for x in self.labels]
        new_df = pd.concat(counts, axis=1).fillna(0).astype(int)
        return new_df

    @property
    def labels(self):
        return self.entries.columns[3:].values.tolist()

    @staticmethod
    def train():
        CxrDataset.transforms = train_transforms

    @staticmethod
    def eval():
        CxrDataset.transforms = test_transforms

