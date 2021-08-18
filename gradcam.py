"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
@Modified by Doyun Kim @ June 10th, 2019
"""
from PIL import Image
import numpy as np
import os
import copy
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.cm as mpl_color_map


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path, dims=None):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)

    if dims:
        resized = im.resize(dims)
        resized.save(path)
    else:
        im.save(path)


def save_class_activation_images(org_img, activation_map, out_dir, file_name, hmp_dims):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """

    if not os.path.exists(str(out_dir)):
        os.makedirs(str(out_dir))

    if (-1 in activation_map):
        # Grayscale activation map
        org_np, heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'jet')
        # Save colored heatmap
        #path_to_file = os.path.join('./results', file_name+'_Cam_Heatmap.png')
        #save_image(heatmap, path_to_file)
        # Save heatmap on iamge
        #path_to_file = os.path.join(out_dir, file_name+'_heatmap.png')
        #save_image(org_np, path_to_file)
        # SAve grayscale heatmap
        path_to_file = os.path.join(out_dir, file_name+'_org.png')
        save_image(org_np, path_to_file, dims=hmp_dims)

    else:
        # Grayscale activation map
        org_np, heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'jet')
        # Save colored heatmap
        #path_to_file = os.path.join('./results', file_name+'_Cam_Heatmap.png')
        #save_image(heatmap, path_to_file)
        # Save heatmap on iamge
        path_to_file = os.path.join(out_dir, file_name+'_hmp.png')
        save_image(heatmap_on_image, path_to_file, dims=hmp_dims)
        # SAve grayscale heatmap
        path_to_file = os.path.join(out_dir, file_name+'_org.png')
        save_image(org_np, path_to_file, dims=hmp_dims)

    return np.array(heatmap.resize(hmp_dims))


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """

    if (-1 in activation):
        zero_activation = np.zeros((org_im.cpu().numpy().shape[2], org_im.cpu().numpy().shape[3]))
        # Get colormap
        color_map = mpl_color_map.get_cmap(colormap_name)
        no_trans_heatmap = color_map(zero_activation)
        # Change alpha channel in colormap to make sure original image is displayed
        heatmap = copy.copy(no_trans_heatmap)
        heatmap[:, :, 3] = 0.2
        heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
        no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

        # Apply heatmap on iamge
        org_np = org_im.cpu().numpy()
        org_np = (org_np - np.amin(org_np))
        org_np = org_np / np.amax(org_np)
        org_np = np.squeeze(org_np, axis=0)
        org_np = np.transpose(org_np, (1, 2, 0))
        # need to change rgb format
        org_np = np.squeeze(np.stack((org_np,) * 3, -1))
        org_pil = Image.fromarray((org_np*255).astype(np.uint8))

        heatmap_on_image = org_np
    else:
        # Get colormap
        color_map = mpl_color_map.get_cmap(colormap_name)
        no_trans_heatmap = color_map(activation)
        # Change alpha channel in colormap to make sure original image is displayed
        heatmap = copy.copy(no_trans_heatmap)
        heatmap[:, :, 3] = 0.2
        heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
        no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

        # Apply heatmap on iamge
        org_np = org_im.cpu().numpy()
        org_np = (org_np - np.amin(org_np))
        org_np = org_np / np.amax(org_np)
        org_np = np.squeeze(org_np, axis=0)
        org_np = np.transpose(org_np, (1, 2, 0))
        # need to change rgb format
        org_np = np.squeeze(np.stack((org_np,) * 3, -1))
        org_pil = Image.fromarray((org_np*255).astype(np.uint8))

        heatmap_on_image = Image.new("RGBA", org_pil.size)
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_pil.convert('RGBA'))
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)

    return org_np, no_trans_heatmap, heatmap_on_image


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, model_type):
        self.model = model
        self.type = model_type
        self.gradients = None
        self._prob = nn.Sigmoid()

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None

        #for module_pos, module in self.model.features._modules.items():
        if self.type == 0:
            x = self.model.module.main.features(x)
            x = F.relu(x)
        elif self.type == 1:
            x = self.model.main.features(x)
        elif self.type == 3:
            x = self.model.module.main.features(x)
        else:
            x1 = self.model.abn_fe.main.features(x)
            x2 = self.model.tb_fe.main.features(x)
            x = torch.cat((x1, x2), dim=1)

        x.register_hook(self.save_gradient)
        conv_output = x
        return conv_output, x

    def forward_pass(self, x, z=None):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)

        if self.type == 0:
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            # logit output
            x = self.model.module.main.classifier(x)
            #x = self.model.main.classifier(x)
        elif self.type == 1:
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            # logit output
            x = self.model.main.classifier(x)
        elif self.type == 3:
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            # logit output
            x = torch.cat((x, z), dim=1)
            x = self.model.module.main.classifier(x)
        else:
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)

            x_s = torch.chunk(x, 2, dim=1)
            x1 = self.model.abn_fe.main.classifier(x_s[0])
            x2 = self.model.tb_fe.main.classifier(x_s[1])
            x = torch.cat((F.sigmoid(x1), F.sigmoid(x2)), dim=1)
            x = self.model.tb_cls(x)

        # prob output
        y = self._prob(x)

        return conv_output.cpu(), x.cpu(), y.cpu()


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, model_type):
        self.model = model
        self.model.eval()
        self.type = model_type
        self.extractor = CamExtractor(self.model, model_type)

    def generate_cam(self, input_image, input_txt=None, target_class=None, cam_w=None, ens_flg=False, ens_cam=None, th_cam=0.0):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        if ens_flg == False:
            if self.type == 3:
                conv_output, model_output, prob_output = self.extractor.forward_pass(input_image, input_txt)
            else:
                conv_output, model_output, prob_output = self.extractor.forward_pass(input_image)

            if target_class is None:
                target_class = np.argmax(model_output.data.numpy())

            # Target for backprop
            one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
            one_hot_output[0][target_class] = 1
            # Zero grads
            self.model.zero_grad()

            # Backward pass with specified target
            model_output.backward(gradient=one_hot_output, retain_graph=True)
            # Get hooked gradients
            guided_gradients = self.extractor.gradients.cpu().data.numpy()[0]
            # Get convolution outputs
            target = conv_output.data.numpy()[0]
            # Get weights from gradients
            weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
            # Create empty numpy array for cam
            cam = np.zeros(target.shape[1:], dtype=np.float32)
            # Multiply each weight with its conv output and then, sum
            for i, w in enumerate(cam_w):
                #if i >= 1024:
                #    w = 0.0
                cam += w * target[i, :, :]

            cam = np.maximum(cam, 0)
            if np.max(cam) == np.min(cam):
                cam = np.zeros(cam.shape)  # Normalize between 0-1
            else:
                cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1

            #review_cam
            cam[cam < th_cam] = 0
        else:
            conv_output, model_output, prob_output = self.extractor.forward_pass(input_image)
            cam = ens_cam

        if np.max(cam) == np.min(cam):
            cam = np.zeros(cam.shape)  # Normalize between 0-1
        else:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam_org = cam

        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255

        return cam, prob_output[0, target_class], target_class, cam_org


#if __name__ == '__main__':
#    # Get params
#    target_example = 0  # Snake
#    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
#        get_example_params(target_example)
#    # Grad cam
#    grad_cam = GradCam(pretrained_model, target_layer=11)
#    # Generate cam mask
#    cam = grad_cam.generate_cam(prep_img, target_class)
#    # Save mask
#    save_class_activation_images(original_image, cam, file_name_to_export)
#    print('Grad cam completed')
