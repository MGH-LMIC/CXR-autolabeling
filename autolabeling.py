import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map

from tqdm import tqdm
from pathlib import Path
from prettytable import PrettyTable
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_curve, precision_recall_curve

import torch
import torchnet as tnt
import torch.nn.functional as F

from utils import logger
from environment import TestEnvironment, initialize, print_label_name
from gradcam import GradCam, save_class_activation_images
from data import CxrDataset, EXT_DATA_BASE
from atlasmethod import EX_AI

import time

ATLAS_GEN = False
atlas_name = 'cardiomegaly'
# 'cardiomegaly', 'atelectasis', 'pulmonary_edema', 'pneumonia', 'pleural_effusion'

class Tester:
    def __init__(self, env, pt_runtime="test", fn_net=None, fl_gradcam=False, cls_gradcam=None, id_prob=None, fl_ensemble=False, fl_exai=False, f_name='sim', f_csv=None):
        self.env = env
        self.pt_runtime = pt_runtime
        self.fl_prob = False if id_prob == None else True
        self.id_prob = id_prob
        self.f_name = f_name
        self.fl_ensemble = fl_ensemble
        # for multiple class and binary label tasks
        self.pf_metric = {
                'loss': [],
                'accuracy': [],
                'sensitivity': [],
                'specificity': [],
                'auc_score': [],
                'ap_score': [],
                'mse_score': []
        }
        self.fn_net = fn_net
        self.fl_gradcam = fl_gradcam
        self.cls_gradcam = cls_gradcam
        self.th_gradcam = 0.5
        self.fl_gradcam_save = True

        #explainable methods
        self.fl_exai = fl_exai
        if self.fl_exai:
            self.fl_gradcam = True
            self.cls_gradcam = [
                        'Hilar/mediastinum>Cardiomegaly>.',
                        'Lung density>Increased lung density>Atelectasis',
                        'Lung density>Increased lung density>Pulmonary edema',
                        'Lung density>Increased lung density>pneumonia',
                        'Pleura>Pleural effusion>.'
                    ]
            self.th_gradcam = 0.5
            self.ex_method = EX_AI(env, pt_runtime=pt_runtime, thr=0.5, f_name=f_name, ext_data_csv=f_csv)

    def load(self):
        pt_file = self.pt_runtime.joinpath(f'train.pkl')
        with open(pt_file, 'rb') as f:
            self.pf_metric = pickle.load(f)

    def test_evaluation(self, epoch=1, fl_save=False):
        if self.fn_net == None:
            pt_model = self.pt_runtime.joinpath(f'model_epoch_{epoch:04d}.pth.tar')
        else:
            pt_model = self.pt_runtime.joinpath(str(self.fn_net))

        self.env.load_model(pt_model)

        try:
            self.load()
        except:
            logger.debug('there is no pkl to load.')

        _, _, _ = self.test(epoch, self.env.test_loader, fl_save=fl_save)

        if False:
            self.algorithm_attribution(self.env.gradcam_loader)

        if self.fl_gradcam:
            _, _, _ = self.gradcam_data(self.env.gradcam_loader)


    def test_ensemble_evaluation(self, epoch=1, fl_save=False, n_ens=1):

        predict = []
        target  = []

        if self.fl_gradcam:
            cams = np.ones((len(self.env.gradcam_loader), len(self.cls_gradcam), 16, 16))

        if ATLAS_GEN:
            gradcam_df = pd.DataFrame(columns=[f'{x:03d}' for x in range(256)])

        for k in range(n_ens):
            pt_model = self.pt_runtime.joinpath(str(self.fn_net)+f'_{k:02d}.pth.tar')
            self.env.load_model(pt_model)

            #logger.info(f'network to test: {self.env.model}')
            try:
                self.load()
            except:
                logger.debug('there is no pkl to load.')

            _, pred, tar = self.test(epoch, self.env.test_loader, fl_save=False)

            predict.append(pred)
            target.append(tar)

        # evaluate ensemble's performance
        prob_ens = self.ensemble_performance(predict, target, n_ens, fl_save=fl_save)

        if self.fl_exai:
            prob_in = pd.DataFrame(prob_ens.cpu().numpy()[:,1:])
            prob_in['PATH'] = self.env.test_loader.dataset.entries['PATH']
            self.ex_method.input_preparation(prob_in)

        if self.fl_gradcam:
            cams = np.ones((len(self.env.gradcam_loader), len(self.cls_gradcam), 16, 16))
            for k in range(n_ens):
                pt_model = self.pt_runtime.joinpath(str(self.fn_net)+f'_{k:02d}.pth.tar')
                self.env.load_model(pt_model)

                start = time.time()
                _, _, cam = self.gradcam_data(self.env.gradcam_loader, prob_ens=prob_ens)
                #review_cam
                #cams *= cam
                cams += cam
                end = time.time()
                print(f'{k:02d} model gradcam time: {end-start} sec')

            _, _, cams = self.gradcam_data(self.env.gradcam_loader, ens_flg=True, cams_ens=cams, prob_ens=prob_ens)

        if self.fl_exai:
            start = time.time()
            self.ex_method.run(cams)
            end = time.time()
            print(f'self-annotation time: {end-start} sec')

        if ATLAS_GEN:
            for k in range(len(self.env.gradcam_loader)):
                gradcam_df.loc[k] = cams[k].flatten()
                print(f"[{atlas_name}]Atlas generation: {k:5d}")

            gradcam_df['PATH'] = self.env.gradcam_loader.dataset.entries['PATH']
            gradcam_df.to_csv(self.pt_runtime.joinpath(f'gradcam_atlas_{atlas_name}.csv'), index=False)


    def ensemble_performance(self, predict, target, n_ens, fl_save=False):

        pred_ens = torch.zeros(predict[0].shape).to(self.env.device)
        #pred_ens = np.zeros(predict[0].shape)
        for i in range(n_ens):
            pred_ens += torch.from_numpy(predict[i]).to(self.env.device)

        pred_ens /= n_ens
        targ_ens =  torch.from_numpy(target[0]).to(self.env.device)

        aucs, aps = self.AUC_AP_metric(pred_ens, targ_ens)
        correct, total = self.ACC_metric(pred_ens, targ_ens)
        self.Per_print(correct=correct, total=total, aucs=aucs, aps=aps)

        if fl_save:
            test_set = self.env.test_loader.dataset
            labels = self.env.labels
            self.roc_evaluation(test_set, pred_ens, targ_ens, labels)

        return pred_ens


    def AUC_AP_metric(self, output, target):
        out_dim = output.shape[1]
        aucs = [tnt.meter.AUCMeter() for i in range(out_dim)]
        aps = [tnt.meter.APMeter() for i in range(out_dim)]

        for i in range(out_dim):
            mask_out, mask_tar = self.mask_pred(output[:, i], target[:, i])
            try:
                aucs[i].add(mask_out, mask_tar)
                aps[i].add(mask_out, mask_tar)
            except:
                continue

        return aucs, aps

    def MSE__metric(self, output, target):
        out_dim = 1
        mses = [tnt.meter.MSEMeter() for i in range(out_dim)]

        mses[0].add(output[:, -1], target[:, -1])

        return mses

    def ACC_metric(self, output, target):
        mask_out, mask_tar = self.mask_pred(output, target)

        ones = torch.ones(mask_out.shape).int().to(self.env.device)
        zeros = torch.zeros(mask_out.shape).int().to(self.env.device)

        pred = torch.where(mask_out > 0.5, ones, zeros)
        correct = pred.eq(mask_tar.int()).sum().item()
        total = len(mask_tar)

        return correct, total


    def Per_print(self, correct=None, total=None, aucs=None, aps=None, mses=None):
        labels = self.env.labels

        out_dim = len(aucs)

        percent = 100. * correct / total

        logger.info(f"accuracy {correct}/{total} "
                    f"({percent:.2f}%)")

        p = PrettyTable()
        p.field_names = ["findings", "auroc score", "ap score"]
        auc_cnt = out_dim
        for i in range(out_dim):
            try:
                #p.add_row([labels[i], f"{aucs[i].value()[0]:.4f}", f"{aps[i].value()[0]:.4f}"])
                p.add_row([f'E-{labels[i]}', f"{aucs[i].value()[0]:.4f}", f"{aps[i].value()[0]:.4f}"])
            except:
                p.add_row([labels[i], "-", "-"])

        try:
            list_aucs=[]
            for k in aucs:
                if type(k.value()) == tuple:
                    if np.isnan(k.value()[0]) == False:
                        list_aucs.append(k.value()[0])

            list_aps=[]
            for k in aps:
                if type(k.value()) == torch.Tensor:
                    if np.isnan(k.value()[0]) == False:
                        list_aps.append(k.value()[0])

            ave_auc = np.mean(list_aucs)
            ave_ap  = np.mean(list_aps)
            tbl_str = p.get_string(title=f"Ensemble-performance (avg auc {ave_auc:.4f}, mean ap {ave_ap:.4f})")
            logger.info(f"\n{tbl_str}")
        except:
            print("We cannot calcuate average acu scores")
            ave_auc = 0
            ave_ap = 0


    def test(self, epoch, test_loader, fl_save=False):
        test_set = test_loader.dataset
        out_dim = self.env.out_dim
        labels = self.env.labels

        aucs = [tnt.meter.AUCMeter() for i in range(out_dim)]
        aps = [tnt.meter.APMeter() for i in range(out_dim)]

        CxrDataset.eval()
        self.env.model.eval()

        with torch.no_grad():
            correct = 0
            total = 0

            predict_seq = torch.FloatTensor().to(self.env.device)
            target_seq = torch.FloatTensor().to(self.env.device)

            tqdm_desc = f'testing '
            t = tqdm(enumerate(test_loader), total=len(test_loader), desc=tqdm_desc,
                    dynamic_ncols=True)

            for bt_idx, tp_data in t:
                output, target = self.test_batch(tp_data)

                # Network outputs
                predict_seq = torch.cat((predict_seq, F.sigmoid(output)), dim=0)
                target_seq = torch.cat((target_seq, target), dim=0)

                for i in range(out_dim):
                    mask_out, mask_tar = self.mask_pred(output[:, i], target[:, i])
                    try:
                        aucs[i].add(mask_out, mask_tar)
                        aps[i].add(mask_out, mask_tar)
                    except:
                        continue

                mask_out, mask_tar = self.mask_pred(output, target)

                ones = torch.ones(mask_out.shape).int().to(self.env.device)
                zeros = torch.zeros(mask_out.shape).int().to(self.env.device)

                pred = torch.where(mask_out > 0., ones, zeros)
                correct += pred.eq(mask_tar.int()).sum().item()
                total += len(mask_tar)
                #pred = torch.where(output > 0., ones, zeros)
                #correct += pred.eq(target.int()).sum().item()

            #total = len(test_loader.sampler) * out_dim
            percent = 100. * correct / total

            logger.info(f"val epoch {epoch:03d}:  "
                        f"accuracy {correct}/{total} "
                        f"({percent:.2f}%)")

            p = PrettyTable()
            p.field_names = ["findings", "auroc score", "ap score"]
            auc_cnt = out_dim
            for i in range(out_dim):
                try:
                    p.add_row([labels[i], f"{aucs[i].value()[0]:.4f}", f"{aps[i].value()[0]:.4f}"])
                except:
                    p.add_row([labels[i], "-", "-"])

            if fl_save:
                self.roc_evaluation(test_set, predict_seq, target_seq, labels)

            if self.fl_prob:
                self.df_prob = pd.DataFrame()
                self.df_prob['PATH_CHECK'] = test_set.entries['PATH']
                self.df_prob['PROB'] = predict_seq.cpu().numpy()[:, self.id_prob]

            try:
                list_aucs=[]
                for k in aucs:
                    if type(k.value()) == tuple:
                        if np.isnan(k.value()[0]) == False:
                            list_aucs.append(k.value()[0])

                list_aps=[]
                for k in aps:
                    if type(k.value()) == torch.Tensor:
                        if np.isnan(k.value()[0]) == False:
                            list_aps.append(k.value()[0])
                        
                ave_auc = np.mean(list_aucs)
                ave_ap  = np.mean(list_aps)

                tbl_str = p.get_string(title=f"performance (avg auc {ave_auc:.4f}, mean ap {ave_ap:.4f})")
                logger.info(f"\n{tbl_str}")
            except:
                print("We cannot calcuate average auc scores")
                ave_auc = 0
                ave_ap = 0

        self.pf_metric[f'accuracy'].append((epoch, correct / total))
        self.pf_metric[f'auc_score'].append((epoch, ave_auc))
        self.pf_metric[f'ap_score'].append((epoch, ave_ap))

        return ave_auc, predict_seq.cpu().numpy(), target_seq.cpu().numpy()

    def mask_pred(self, output, target):
        mask_one = torch.ones(output.shape, dtype=torch.uint8, device=self.env.device)
        mask_zero = torch.zeros(output.shape, dtype=torch.uint8, device=self.env.device)

        #mask = torch.where(target == -1, mask_zero, mask_one)
        mask = torch.where(target == -1, mask_zero, mask_one).bool()
        mask_output = output.masked_select(mask.to(self.env.device))
        mask_target = target.masked_select(mask.to(self.env.device))

        return mask_output, mask_target

    def test_batch(self, tp_data, fl_input=False):
        # to support different types of models.
        if self.env.type == 0:
            data = tp_data[0]
            target = tp_data[1]
            info = tp_data[2]
            data, target, info = data.to(self.env.device), target.to(self.env.device), info.to(self.env.device)
            #data, target = data.to(self.env.device), target.to(self.env.device)
            #network output
            output = self.env.model(data)
        elif self.env.type == 1:
            data1 = tp_data[0]
            data2 = tp_data[1]
            target = tp_data[2]
            data1, data2, target = data1.to(self.env.device), data2.to(self.env.device), target.to(self.env.device)
            #network output
            output = self.env.model(data1, data2)
        elif self.env.type == 3:
            data = tp_data[0]
            target = tp_data[1]
            info = tp_data[2]
            data, target, info = data.to(self.env.device), target.to(self.env.device), info.to(self.env.device)
            #network output
            output = self.env.model(data, info)

        if fl_input == False:
            return output, target
        else:
            return data, info, output


    def gradcam_data(self, test_loader, hmp_dims=(512,512), ens_flg=False, cams_ens=None, prob_ens=None):
        # threshold to draw a heatmap
        out_dim = self.env.out_dim

        CxrDataset.eval()
        self.env.model.eval()
        #with torch.no_grad():
        gradcam_res_list  = []
        gradcam_path_list = []

        cams = np.zeros((len(test_loader), len(self.cls_gradcam), 16, 16))

        grad_cam = GradCam(self.env.model, self.env.type)
        for batch_idx, (data, target, info) in enumerate(test_loader):
            #data, target = data.to(self.env.device), target.to(self.env.device)
            data, target, info = data.to(self.env.device), target.to(self.env.device), info.to(self.env.device)
            # Grad CAM
            #grad_cam = GradCam(self.env.model, self.env.type)
            if self.cls_gradcam == None:
                gradcam_res, gradcam_path = self.gradcam_save_maxcls(grad_cam, data, test_loader, batch_idx, hmp_dims, info)
            else:
                if self.fl_ensemble:
                    cam = self.gradcam_save_argcls_ens(grad_cam, data, test_loader, batch_idx, hmp_dims, info, ens_flg=ens_flg, cams_ens=cams_ens, prob_ens=prob_ens)
                else:
                    gradcam_res, gradcam_path = self.gradcam_save_argcls(grad_cam, data, test_loader, batch_idx, hmp_dims, info)

            try:
                if self.fl_ensemble:
                    cams[batch_idx, :, :, :] = cam
                else:
                    gradcam_res_list.append(gradcam_res.tolist())
                    gradcam_path_list.append(gradcam_path)

            except AttributeError as e:
                print("No GradCam result?")

        if False:
            self.gradcam_thumbnail()


        return gradcam_res_list, gradcam_path_list, cams

    def gradcam_save_maxcls(self, grad_cam, data, test_loader, batch_idx, hmp_dims, info):
        if self.env.type == 3:
            cam, prob, tcls = grad_cam.generate_cam(data, info)
        else:
            cam, prob, tcls = grad_cam.generate_cam(data)

        noPlotflg = np.array([-1])
        # when we draw gradcam, we have to batch size as 1.
        file_name = test_loader.dataset.entries['PATH'][batch_idx]
        path_name = file_name.split(".")[0]

        if prob >= self.th_gradcam:
            target_class = self.env.labels[tcls]
            label_list = re.split(' \- |\/| ', target_class)
            label_name = "_".join(label_list)
            path_name = "_".join([path_name, label_name])

            cam_rs = save_class_activation_images(data, cam, self.pt_runtime.joinpath(f"gradcam_image"), path_name, hmp_dims)
            return cam_rs, path_name
        else:
            cam_rs = save_class_activation_images(data, noPlotflg, self.pt_runtime.joinpath("gradcam_image"), path_name, hmp_dims)
            return None, None

    def gradcam_save_argcls(self, grad_cam, data, test_loader, batch_idx, hmp_dims, info):

        if self.cls_gradcam[0] == 'all':
            self.cls_gradcam = self.env.labels

        for i, nm_tcls in enumerate(self.cls_gradcam):
            ## need to implement to find index among self.env.labels from string of target class
            ## code start here!!!!
            id_tcls = self.env.labels.index(nm_tcls)
            if self.env.type == 3:
                cam, prob, tcls = grad_cam.generate_cam(data, info, target_class=id_tcls)
            else:
                cam_w = self.env.model.module.main.classifier.weight[id_tcls].cpu().detach().numpy()
                cam, prob, tcls, _ = grad_cam.generate_cam(data, target_class=id_tcls, cam_w=cam_w)
            noPlotflg = np.array([-1])
            # when we draw gradcam, we have to batch size as 1.
            file_name = test_loader.dataset.entries['PATH'][batch_idx]
            path_name = file_name.split(".")[0]

            target_class = self.env.labels[tcls]
            label_list = re.split(' \- |\/| ', target_class)
            label_name = "_".join(label_list)
            label_name = label_name.strip('>.').split('>')[-1]
            #path_name = "_".join([f'{int(prob*1000):04d}', path_name, label_name])

            if prob >= self.th_gradcam:
                cam_rs = save_class_activation_images(data, cam, self.pt_runtime.joinpath(f"gradcam_image_{label_name}"), path_name, hmp_dims)

            cam_list=[]
            path_list=[]

            path_list.append(path_name)
        return cam_list, path_list

    def gradcam_save_argcls_ens(self, grad_cam, data, test_loader, batch_idx, hmp_dims, info, ens_flg=False, cams_ens=None, prob_ens=None):

        if self.cls_gradcam[0] == 'all':
            self.cls_gradcam = self.env.labels

        cams = np.zeros((len(self.cls_gradcam), 16, 16))
        for i, nm_tcls in enumerate(self.cls_gradcam):
            ## need to implement to find index among self.env.labels from string of target class
            ## code start here!!!!
            id_tcls = self.env.labels.index(nm_tcls)
            cam_w = self.env.model.module.main.classifier.weight[id_tcls].cpu().detach().numpy()
            if prob_ens[batch_idx, id_tcls].item() >= self.th_gradcam:
                if ens_flg == True:
                    cam, prob, tcls, cam_low = grad_cam.generate_cam(data, target_class=id_tcls, cam_w=cam_w, ens_flg=True, ens_cam=cams_ens[batch_idx, i, :, :])
                    cams[i, :, :] = cam_low

                    noPlotflg = np.array([-1])
                    # when we draw gradcam, we have to batch size as 1.
                    file_name = test_loader.dataset.entries['PATH'][batch_idx]
                    path_name = file_name.split(".")[0]

                    label_name = print_label_name[tcls]

                    if ATLAS_GEN:
                        label_name = f"ATLAS_{atlas_name}"

                    #if prob_ens[batch_idx, id_tcls].item() >= self.th_gradcam:
                    if ATLAS_GEN:
                        cam_rs = save_class_activation_images(data, cam, self.pt_runtime.joinpath(f"{label_name}"), path_name, hmp_dims)
                    else:
                        if '/' in path_name:
                            self.pt_runtime.joinpath(f"explain_sample/{self.f_name}/{label_name}/{path_name}").parent.mkdir(parents=True, exist_ok=True)
                        cam_rs = save_class_activation_images(data, cam, self.pt_runtime.joinpath(f"explain_sample/{self.f_name}/{label_name}"), path_name, hmp_dims)
                else:
                    #review_cam
                    cam, prob, tcls, cam_low = grad_cam.generate_cam(data, target_class=id_tcls, cam_w=cam_w, th_cam=0.5)
                    cams[i, :, :] = cam_low

        return cams

    def roc_evaluation(self, test_set, predict_seq, target_seq, labels):
        out_dim = self.env.out_dim
        df_data = pd.DataFrame()
        df_data['PATH'] = test_set.entries['PATH']
        for i in range(out_dim):
            df_data[f'{labels[i]}'] = predict_seq.cpu().numpy()[:, i]
            df_data[f'{labels[i]}_GT'] = target_seq.cpu().numpy()[:, i]

        t = self.pt_runtime.joinpath('roc_result')
        Path.mkdir(t, parents=True, exist_ok=True)
        df_data.to_excel(t.joinpath('save_predicted_probabilities.xlsx'))

        roc_dim = out_dim 
        for i in range(roc_dim):
            mask_out, mask_tar = self.mask_pred(predict_seq[:, i], target_seq[:, i])
            if mask_tar.cpu().numpy().size != 0 :
                fpr, tpr, thresholds = roc_curve(mask_tar.cpu().numpy(), mask_out.cpu().numpy())
                pre, rec, thresholds_pr = precision_recall_curve(mask_tar.cpu().numpy(), mask_out.cpu().numpy())
                #logger.debug(f"{predict_seq.cpu().numpy()}")
                df = pd.DataFrame()
                df[f'specificity'] = 1-fpr
                df[f'sensitivity'] = tpr
                df[f'thresholds'] = thresholds

                label_name = print_label_name[i]
                df.to_excel(t.joinpath(f'save_{i:03d}_{label_name}_sensitivity_specificity.xlsx'))
                del df

                if False:
                    # ROC plot
                    fig, (ax1, ax2) = plt.subplots(1,2)
                    ax1.plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'ROC curve')
                    ax1.set_title(f'ROC curve for {labels[i]}')
                    ax1.set(xlabel='False positive rate', ylabel='True positive rate')
                    # PR plot
                    ax2.plot(rec, pre, color = 'darkorange', lw = 2, label = 'Precision-Recall curve')
                    ax2.set_title(f'Precision-Recall curve')
                    ax2.set(xlabel='Recall', ylabel='Precision')
                    plt.savefig(t.joinpath(f'{i:03d}_{label_name}_curve.png'))
                else:
                    # ROC plot
                    fig, ax1 = plt.subplots(1,1)
                    ax1.plot(fpr, tpr, color = 'darkorange', lw = 2, label = f'{label_name}')
                    ax1.set_title(f'ROC curve for {label_name}')
                    ax1.set(xlabel='False positive rate', ylabel='True positive rate')
                    plt.savefig(t.joinpath(f'{i:03d}_{label_name}_curve.png'))


    def save_prob(self, input_file, save_path):
        df = pd.read_csv(input_file)
        df.insert(6, 'prob', self.df_prob.PROB)
        df.insert(6, 'path_check', self.df_prob.PATH_CHECK)

        df.to_excel(save_path)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Testng Our Explainable AI Model on CXR")
    parser.add_argument('--cuda', default=None, type=str, help="use GPUs with its device ids, separated by commas")

    args = parser.parse_args()
    args.in_dim      = 1
    args.out_dim     = 21
    args.labels      = None
    args.paths       = None
    args.runtime_dir = 'autolabeling'
    args.type        = 0
    args.pretr_net   = 'pa_feat_model'
    args.gradcam     = False
    args.gradcam_cls = None
    args.fl_save     = False
    args.id_prob     = None
    args.test_csv    = 'autolabeling_5_features_490_cases.csv'
    args.arch        = None
    args.Nens        = 6
    args.exai        = True
    args.simname     = 'Outputs'
    args.seed        = -1

    runtime_path, device = initialize(args, fl_demo=True)
    fl_ensemble = False if args.Nens == 1 else True

    # start training
    env = TestEnvironment(device, mtype=args.type, in_dim=args.in_dim, out_dim=args.out_dim, name_labels=args.labels, name_paths=args.paths, testset_csv=args.test_csv, name_model=args.arch, r_seed=args.seed)
    t = Tester(env, pt_runtime=runtime_path, fn_net=args.pretr_net, fl_gradcam=args.gradcam, cls_gradcam=args.gradcam_cls, id_prob=args.id_prob, fl_ensemble=fl_ensemble, fl_exai=args.exai, f_name=args.simname, f_csv=args.test_csv)

    if(fl_ensemble):
        t.test_ensemble_evaluation(fl_save=args.fl_save, n_ens = args.Nens)
    else:
        t.test_evaluation(fl_save=args.fl_save)

