import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import shutil
import dataframe_image as dfi
import seaborn as sns
import scipy.stats as stats
from scipy import ndimage
from PIL import Image
import cv2
import time

from data import EXT_DATA_BASE, label_name

import warnings
warnings.filterwarnings("ignore")

color_dict = dict({
       'Basis patches':'tab:gray',
       'cardiomegaly':'tab:purple',
       'atelectasis':'tab:green',
       'pulmonary_edema':'tab:orange',
       'pneumonia':'tab:red',
       'other_int_opacity':'tab:cyan', 
       'decreased_lung_volume':'tab:brown',
       'pleural_effusion':'tab:blue',
       'fracture':'tab:purple',
       'pneumothorax':'tab:brown',
       'ap':'tab:blue',
       'pa':'tab:red',
       'F': 'tab:blue',
       'M': 'tab:red',
       'group 0': 'tab:blue',
       'group 1': 'tab:red',
       'group 2': 'tab:green',
       'case_patches': 'tab:red',
       'test_patches': 'tab:red'
    })
feat_points = dict({
    'cardiomegaly': [(-3,6), (-2,4), (6,4), (10,10), (15,9), (15,12)],
    'atelectasis': [(2,8), (2,-1), (7,-1), (9.5,4)],
    'pulmonary_edema': [(2,15), (6,13), (5,10), (6,6)],
    'pneumonia': [(4,10), (3,3), (10,4), (10,10)],
    'other_int_opacity': [(3.5,0.7), (8,3), (6,8.6), (5,10.5)],
    'decreased_lung_volume': [(6,9), (8,8), (10,8), (16,11), (14,2), (15,-1)],
    'pleural_effusion': [(2,-1.5), (2,7.5), (12,1), (7.5,3)],
    'pa': [(-1,6), (3,6), (7.5,3), (12,1), (12,4), (15,6), (14,8.5), (10,11)],
    'ap': [(0,6), (4,7), (8,9), (9,6), (10,3), (14,3), (13,6), (12,9.5)]
    })

S_LABELS = ['PATH',
       'Hilar/mediastinum>Cardiomegaly>.',
       'Lung density>Increased lung density>Atelectasis',
       'Lung density>Increased lung density>Pulmonary edema',
       'Lung density>Increased lung density>pneumonia',
       'Pleura>Pleural effusion>.'
       ]
S_NAMES = ['PATH',
       'cardiomegaly',
       'atelectasis',
       'pulmonary_edema',
       'pneumonia',
       'pleural_effusion'
       ]
S_pSIM_TH = [0.15, 0.60, 0.55, 0.65, 0.20] 

class EX_AI():
    def __init__(self, env, pt_runtime=None, thr=0.5, n_ens=6, dset='train', th_cor=0.2, f_name=None, ext_data_csv=None):
        self.thr = thr
        self.env = env
        self.n_ens = n_ens
        self.pt_runtime = pt_runtime
        self.f_name = f_name
        self.out_path = self.pt_runtime.joinpath(f'{self.f_name}')
        self.out_path.mkdir(parents=True, exist_ok=True)

        self.view = self.env.labels[0]

        self.database = pd.read_excel(self.pt_runtime.joinpath(f'ATLAS_db/{dset}_predicted_probabilities.xlsx'))
        self.df_mdist = {}
        for cls in S_NAMES[1:]:
            if self.pt_runtime.joinpath(f'ATLAS_db/mDist_atlas_{cls}.csv').exists():
                self.df_mdist[f'{cls}'] = pd.read_csv(self.pt_runtime.joinpath(f'ATLAS_db/mDist_atlas_{cls}.csv'))
            else:
                self.df_mdist[f'{cls}'] = []

        self.umap = pickle.load(open(self.pt_runtime.joinpath(f'ATLAS_db/umap-model-type0.pkl'), 'rb'))
        self.embedded = pd.read_csv(self.pt_runtime.joinpath(f'ATLAS_db/umap-data-type0.csv'))
        self.case_embedded = pd.DataFrame(columns=['label', '0', '1', 'confidence', 'patch_similarity'])

        self.fl_ext_data = True
        df_ext_data = pd.read_csv(EXT_DATA_BASE.joinpath(f'{ext_data_csv}'))
        self.df_ext = df_ext_data

    def euclid_dist(self, t1, t2, weight=[]):
        if len(weight)>0:
            weight = np.asarray(weight)
            tmp = weight*(t1-t2)**2
            out = np.sqrt(tmp.sum(axis = 1))
        else:
            out = np.sqrt(((t1-t2)**2).sum(axis = 1))
        return out


    def input_preparation(self, prob_test):
        prob_ai = self.database.iloc[:, [1]+[2*x+4 for x in range(20)]]
        
        train = prob_ai.copy(deep=True)
        if 'Unnamed: 0' in train.columns:
            train.drop(columns=['Unnamed: 0'], inplace=True)
        train.reset_index(inplace=True, drop=True)
        train.sort_values(by=['PATH'], inplace=True, ignore_index=True)
        df_tr_path = train.pop('PATH')

        test = prob_test.copy(deep=True)

        self.df_ext = self.df_ext.loc[self.df_ext.PATH.isin(test.PATH)]
        for i, feat in enumerate(label_name):
            self.df_ext[feat] = self.df_ext[feat].astype('float')
            self.df_ext.rename(columns={feat:f'{feat}_gt'}, inplace=True)
            self.df_ext[feat] = 0.0

        for m, row in self.df_ext.iterrows():
            for i, feat in enumerate(label_name):
                self.df_ext.at[m, feat] = test.loc[test.PATH == row.PATH].iloc[0, i]

        ordered_name = ['PATH']
        for m, cls in enumerate(S_NAMES[1:]):
            self.df_ext.rename(columns={S_LABELS[m+1]:f'{cls}_pr', f'{S_LABELS[m+1]}_gt':f'{cls}_gt'}, inplace=True)
            self.df_ext[f'{cls}_ps'] = 0.0    #patch similarity
            self.df_ext[f'{cls}_cf'] = 0.0    #confidence
            self.df_ext[f'{cls}_pSim'] = 0.0  #probability of similarity
            self.df_ext[f'{cls}_agt'] = -1.0   #auto-labeling
            ordered_name.append(f'{cls}_gt')  #original groun truth
            ordered_name.append(f'{cls}_pr')
            ordered_name.append(f'{cls}_ps')
            ordered_name.append(f'{cls}_cf')
            ordered_name.append(f'{cls}_pSim')
            ordered_name.append(f'{cls}_agt')

        self.df_ext = self.df_ext[ordered_name]

        # parameter set-ups
        self.train = train
        self.test = test
        self.df_tr_path = df_tr_path
        self.prob = prob_ai

        for i, col in enumerate(prob_test):
            if col != 'PATH':
                prob_test.rename(columns={i: f"{train.columns.to_list()[i]}"}, inplace=True)
                self.test.rename(columns={i: f"{train.columns.to_list()[i]}"}, inplace=True)
        self.prob_test = prob_test

    def evaluate(self):
        # Select num_images samples with the minimum Euclidean distance
        samples = self.test
        df_samples_path = samples.pop('PATH')

        return samples, df_samples_path

    def get_patch(self, cam, img_path, save_path, fl_merge=False):

        img = np.array(Image.open(str(img_path.resolve())).resize((512,512), Image.ANTIALIAS))
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((img.shape[0],
                       img.shape[1]), Image.ANTIALIAS))

        mask = np.ones(cam.shape)
        mask[cam<100]=0
        img = Image.fromarray(np.uint8(img))

        margin = 40 if fl_merge else 25
        self.do_crop(mask, img, save_path, margin=margin, fl_merge=fl_merge)

    def contourMask(self, image):
        image = image.reshape((image.shape[0], image.shape[1], 1))
        image = np.uint8(image)
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area = np.zeros(len(contours))
        for j in range(len(contours)):
            cnt = contours[j]
            area[j] = cv2.contourArea(cnt)
        mask = np.zeros(image.shape)
        cv2.drawContours(mask, contours, np.argmax(area), 1, -1)#draw largest contour-usually right lung
        temp = np.copy(area[np.argmax(area)])
        area[np.argmax(area)]=0
        cv2.drawContours(mask, contours, np.argmax(area), 2, -1)#draw second largest contour
        contours.clear()
        return np.squeeze(mask, axis=2)

    def do_crop(self, mask, img, img_path, margin=20, fl_merge=False):
        mask = mask if fl_merge else self.contourMask(mask)
        iw, ih = mask.shape
        objs = ndimage.find_objects(mask, 2)
        for m, obj in enumerate(objs):
            if obj != None:
                slice_y, slice_x = obj

                t = max(slice_y.start-margin, 0)
                l = max(slice_x.start-margin, 0)
                b = min(slice_y.stop+margin, ih)
                r = min(slice_x.stop+margin, iw)

                crp_img = img.crop((l, t, r, b))
                res_img = crp_img.resize((128, 128), Image.ANTIALIAS)
                umap_img = crp_img.resize((32, 32), Image.ANTIALIAS)

                path = str(img_path).split('.')[0] + f'_patch_{m:02d}.png'
                res_img.save(path)
                path = str(img_path).split('.')[0] + f'_umap_{m:02d}.png'
                umap_img.save(path)

    def find_umap_transform(self, patch, t_name, save_path, idx, x_bs=100):
        X = cv2.imread(str(patch))[:,:,0].reshape(1, 1024)
        y = self.umap.transform(X)
        self.case_embedded.at[self.case_cnt, ['0', '1']] = y
        self.case_embedded.at[self.case_cnt, ['label']] = t_name
        self.case_cnt += 1
        
        sel_patch = self.embedded.loc[self.embedded.label == t_name]
        sel_patch.reset_index(inplace=True, drop=True)
        dists = self.euclid_dist(y, sel_patch[['0','1']].to_numpy())

        #finding x_bs patchesi & calculating mean distance in embedding space.
        Ntops = np.argsort(dists)[0:(x_bs)]
        basis_paths = sel_patch.PATH.iloc[Ntops].to_list()
            
        return basis_paths, np.mean(dists[Ntops])

    def calculation_ps(self, patch_path, save_path, t_name, path_name=None):
        org_patches = [x for x in patch_path.glob('*org_image_umap*.png')]
        
        min_dists = []
        for m, patch in enumerate(org_patches):
            bas_patches, avg_dist = self.find_umap_transform(patch, t_name, save_path, m)
            min_dists.append(avg_dist)

        #calculating 1-percentile(avg_dist)
        df_mdist = self.df_mdist[f'{t_name}'].copy(deep=True)
        nm_all = df_mdist.shape[0]
        df_mdist.sort_values(by=[f'{t_name}_ps'], ascending=False, inplace=True, ignore_index=True)
        rank_mdist = df_mdist.loc[df_mdist[f'{t_name}_ps']>=np.mean(min_dists)].shape[0]/nm_all
        self.case_embedded.loc[self.case_embedded.label == f'{t_name}', 'patch_similarity'] = rank_mdist*100 if rank_mdist != 0.0 else 0.0001

        shutil.rmtree(patch_path)

        if (t_name !='decreased_lung_volume') & (t_name !='other_int_opacity'):
            df_mdist = self.df_mdist[f'{t_name}'].copy(deep=True)
            nm_all = df_mdist.shape[0]

            df_mdist.sort_values(by=[f'{t_name}_ps'], ascending=False, inplace=True, ignore_index=True)
            rank_mdist = df_mdist.loc[df_mdist[f'{t_name}_ps']>=np.mean(min_dists)].shape[0]/nm_all
            self.df_ext.loc[self.df_ext.PATH.isin([path_name]), f'{t_name}_ps'] = rank_mdist

    def calculate_patch_similarity(self, org_set=None, path=None, t_label=None):
        t_idx = S_LABELS.index(t_label)
        t_name = S_NAMES[t_idx]

        ## need to change for getting input from simulation directly.
        self.ORIG_HM_BASE = self.pt_runtime.joinpath(f'explain_sample/{self.f_name}/{t_name}')

        if t_name in ['cardiomegaly', 'decreased_lung_volume']:
            fl_merge = True
        else:
            fl_merge = False

        try:
            org_path = path

            sample_path = self.out_path.joinpath(org_path.split('.')[0]+f'/cam_{t_name}')
            Path.mkdir(sample_path, parents=True, exist_ok=True)
            patch_path = sample_path.joinpath('patch')
            Path.mkdir(patch_path, parents=True, exist_ok=True)
            shutil.copy(EXT_DATA_BASE.joinpath(org_path), sample_path.joinpath('org_image.png'))
            shutil.copy(self.ORIG_HM_BASE.joinpath(f'{org_path}'.split('.')[0]+'_hmp.png'), sample_path.joinpath('org_image_hmp.png'))

            self.get_patch(org_set[t_idx-1], sample_path.joinpath('org_image.png'), patch_path.joinpath('org_image.png'), fl_merge=fl_merge)

            self.calculation_ps(patch_path, sample_path, t_name, path_name=path)

        except Exception as e: 
            print(e)
            print(f'{path}: No gradcam for {t_name}')


    def run(self, cams):
        # similar group based on features' predicted probability
        samples, df_samples_path = self.evaluate()

        for k, row in samples.iterrows():
            print(f'autolabeling [{k:06d}/{samples.shape[0]:06d}]')
            self.case_cnt = 0
            self.case_embedded = pd.DataFrame(columns=['label', '0', '1', 'confidence', 'patch_similarity']) 
            tmp = row[S_LABELS[1:]]
            hm_classes = tmp[tmp>=self.thr].index.tolist()

            for m, cls in enumerate(hm_classes):
                self.calculate_patch_similarity(org_set=cams[k], path=df_samples_path[k], t_label=cls)

            self.calculate_confidence(tmp, df_samples_path[k])

            if len(hm_classes) > 0:
                rmxs = [rmx for rmx in self.out_path.iterdir() if rmx.is_dir()]
                shutil.rmtree(rmxs[0])

        #Mode selection
        for k, feature in enumerate(S_NAMES[1:]):
            # probability of similarity for positive prediction
            self.df_ext.loc[self.df_ext[f'{feature}_pr']>=0.5, f'{feature}_pSim'] = 2/(1/self.df_ext[f'{feature}_cf'] + 1/(self.df_ext[f'{feature}_ps']))
            # probability of similarity for negative prediction
            self.df_ext.loc[self.df_ext[f'{feature}_pr']<0.5, f'{feature}_pSim']  = self.df_ext.loc[self.df_ext[f'{feature}_pr']<0.5][f'{feature}_cf']

            # auto-labeling in Mode-selection
            self.df_ext.loc[(self.df_ext[f'{feature}_pr']>=0.5)&(self.df_ext[f'{feature}_pSim']>=S_pSIM_TH[k]), f'{feature}_agt'] = 1.0
            self.df_ext.loc[(self.df_ext[f'{feature}_pr']<0.5)&(self.df_ext[f'{feature}_pSim']>=S_pSIM_TH[k]) , f'{feature}_agt'] = 0.0

        # saving auto-labeling results on external dataset.
        self.df_ext.to_csv(self.out_path.joinpath(f'output_autolabeling.csv'))
        shutil.rmtree(self.pt_runtime.joinpath(f'explain_sample'))

    def calculate_confidence(self, prob, path):
        
        pos_cls = prob[prob>=self.thr].index.tolist()
        neg_cls = prob[prob<self.thr].index.tolist()
    
        ai_score = {}
        for k, cls in enumerate(prob.index.to_list()):
            t_pos = self.database.loc[self.database[f'{cls}_GT'] == 1]
            t_neg = self.database.loc[self.database[f'{cls}_GT'] == 0]

            val = prob[cls]
            tmp = t_pos.sort_values(by=[f'{cls}'], ignore_index=True)[cls]
            try:
                p_rank = int(np.round(tmp[tmp >= val].index[0]/tmp.shape[0]*100))
            except:
                p_rank = 100

            tmp = t_neg.sort_values(by=[f'{cls}'], ignore_index=True, ascending=False)[cls]
            try:
                n_rank = int(np.round(tmp[tmp >= val].shape[0]/tmp.shape[0]*100))
            except:
                n_rank = 0

            if cls in pos_cls:
                cof_lvl = max(p_rank-n_rank, 0.0)
                self.case_embedded.loc[self.case_embedded.label == f'{S_NAMES[k+1]}', 'confidence'] = cof_lvl if cof_lvl != 0.0 else 0.0001
            else:
                cof_lvl = max(n_rank-p_rank, 0.0)
                self.case_embedded.loc[self.case_embedded.label == f'{S_NAMES[k+1]}', 'confidence'] = 0

            ai_score[f'{S_NAMES[k+1]}'] = [cof_lvl, p_rank, n_rank, val]
        
            self.df_ext.loc[self.df_ext.PATH.isin([path]), f'{S_NAMES[k+1]}_cf'] = 0.01*ai_score[f'{S_NAMES[k+1]}'][0]


if __name__ == "__main__":
    parser.add_argument('--thr', '-a', default=0.7,  type=float, help='threshold for accuracy')
    args = parser.parse_args()

    exai = EX_AI(thr=args.thr)
    exai.input_preparation()
    if False:
        exai.run()
    if False:
        exai.calculate_confidence()

