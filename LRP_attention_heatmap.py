############################
# only use it to generate attention heatmap
##################first step
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
#torch.cuda.set_device(2)
import time
import numpy as np
import cv2
from samples.CLS2IDX import CLS2IDX
import math
from baselines.ViT.ViT_LRP import deit_base_patch16_224 as vit_base
from baselines.ViT.ViT_LRP import deit_small_patch16_224 as vit_small
from baselines.ViT.ViT_LRP import deit_tiny_patch16_224 as vit_tiny
from baselines.ViT.ViT_LRP import Block
from baselines.ViT.ViT_explanation_generator import LRP

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

# initialize ViT pretrained with DeiT
model = vit_base(pretrained=True).cuda()
model.eval()
attribution_generator = LRP(model)

def print_top_classes(predictions, **kwargs):    
    # Print Top-10 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(CLS2IDX[cls_idx])
        if len(CLS2IDX[cls_idx]) > max_str_len:
            max_str_len = len(CLS2IDX[cls_idx])
    
    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, CLS2IDX[cls_idx])
        output_string += ' ' * (max_str_len - len(CLS2IDX[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)
    # print(class_indices)
    return class_indices

def add_visualization(original_image, class_index=None, start_layer=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index = class_index, start_layer=start_layer).detach()
 
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)

    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')

    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    return transformer_attribution

def find_decisions(orderto, ratio):
    y, i = torch.sort(orderto)
    thre_index = int(196 * ratio)
    thre = y[thre_index]
    return thre
def get_decision(thres, array):
    decision = [[0 for i in range(14)] for i in range(14)]
    for i in range(14):
        for j in range(14):
            temp = np.mean(array[i*16:(i+1)*16,j*16:(j+1)*16])
            if temp <= thres:
                decision[i][j] = 0
            else:
                decision[i][j] = 1
    return decision


def generate_visualization(original_image, class_index=None, start_layer=None):
    i=0
    print(len(original_image))
    for image in original_image:
        for index in class_index:
            temp = add_visualization(image, class_index = index, start_layer=start_layer)
            if i==0 and index == class_index[0]:
                transformer_attribution = temp/len(original_image)
            else:
                transformer_attribution += temp/len(original_image)
        i+=1

    return transformer_attribution


def get_thres(transformer_attribution):

    array = np.array(transformer_attribution) 

    order = []
    for i in range(14):
        for j in range(14):
            temp = np.mean(array[i*16:(i+1)*16,j*16:(j+1)*16]) 
            for k in range(16):
                for q in range(16):
                    array[i*16+k,j*16+q] = temp
            order.append(temp)
            
    orderto = torch.Tensor(order)
    thres1 = find_decisions(orderto, 1-0.7)
    thres2 = find_decisions(orderto, 1-0.7*0.7)
    thres3 = find_decisions(orderto, 1-0.7*0.7*0.7)
    decisions = []
    decision1 = get_decision(thres1, array)
    decisions.append(decision1)
    decision2 = get_decision(thres2, array)
    decisions.append(decision2)
    decision3 = get_decision(thres3, array)
    decisions.append(decision3)
    return decisions
    
    

import torchvision.datasets as datasets
import os
import torch.nn as nn

# Data loading code
traindir = os.path.join('/home/.../datasets/imagenet', 'train')
valdir = os.path.join('/home/.../datasets/imagenet', 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
train_sampler = None
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=192, shuffle=True, 
        num_workers=4, pin_memory=True, sampler=train_sampler)



def genr_decision(model, train_loader):
    decisions = [] 
    transformer_attribution = []

    forimages = []
    for i, (images, target) in enumerate(train_loader):
        if (i+1)%100 == 0:
            try:
                output = model(images.cuda())
                class_top10 = print_top_classes(output)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            print(images.shape)
            for layer in range(12):
                temp= generate_visualization(images, class_index = class_top10, start_layer=layer)
                if i == 99: 
                    transformer_attribution.append(temp)
                else:
                    transformer_attribution[layer] += temp
                    transformer_attribution[layer] /= 2
            del images
            torch.cuda.empty_cache() 
            time.sleep(5)
            if i== 99:
                break
    del model
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return transformer_attribution
        


def generate_masked_image(img, decision):
    tokens = copy.deepcopy(img)
    tokens = tokens.permute(1,2,0)
    tokens_arr = np.array(tokens.cpu())
    for i in range(14):
        for j in range(14):
            if decision[i][j] == 0:
                for k in range(16):
                    for q in range(16):
                        tokens_arr[i*16+k,j*16+q] = (0, 0, 0)  
    tokens = torch.from_numpy(tokens_arr)
    tokens = tokens.permute(2,0,1)
    return tokens

attention_map = genr_decision(model, train_loader)


import sys  
np.set_printoptions(threshold=sys.maxsize)

for l in range(12):
    print("layer:{}".format(l))
    np.save("/home/recordattn_base/layer192_{}.npy".format(l), attention_map[l])

