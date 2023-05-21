# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:10:36 2023

@author: OmerFarukDinc
"""

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.utils import shuffle
# utilization
import os
from tqdm import tqdm
from glob import glob

# data manipulation and visualization tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
# cross-validaion and evaluation tools
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split

# model development and data preparation
import torch
import torchvision

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import SubsetRandomSampler, ConcatDataset
from torch.utils.data import random_split
from torchvision.io import read_image
from torchvision import transforms as t
import pydicom
import os
import math
from matplotlib import image
import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import pydicom
from PIL import Image
def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)
def crop_coords(img):
    """
    Crop ROI from image.
    """
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0).astype("uint8")
    _, breast_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h)

def truncation_normalization(img):
    """
    Clip and normalize pixels in the breast ROI.
    @img : numpy array image
    return: numpy array of the normalized image
    """
    Pmin = np.percentile(img[img!=0], 0.1)
    Pmax = np.percentile(img[img!=0], 99)
    truncated = np.clip(img,Pmin, Pmax)  
    normalized = (truncated - Pmin)/(Pmax - Pmin)
    normalized[img==0]=0
    return normalized

def clahe(img, clip):
    """
    Image enhancement.
    @img : numpy array image
    @clip : float, clip limit for CLAHE algorithm
    return: numpy array of the enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip)
    cl = clahe.apply(np.array(img*255, dtype=np.uint8))
    return cl
def concat_preproces(img1,img2):
    IMG_SIZE=512
    if np.mean(img1/np.max(img1))>0.7:
        img1=(np.max(img1)-img1)
    img1=img1[:,int((img1.shape[1]/3)):]
    (x1, y1, w1, h1) = crop_coords(img1)
    rect1 = patches.Rectangle((x1, y1), w1, h1, linewidth=1, edgecolor='r', facecolor='none')
    img_cropped1 = img1[y1:y1+h1, x1:x1+w1]
    if np.mean(img2/np.max(img2))>0.7:
        img2=(np.max(img2)-img2)
    img2=img2[:,:int((img2.shape[1]/3))]

    (x2, y2, w2, h2) = crop_coords(img2)
    rect2 = patches.Rectangle((x2, y2), w2, h2, linewidth=1, edgecolor='r', facecolor='none')
    img_cropped2 = img2[y2:y2+h2, x2:x2+w2]
    if img_cropped1.size >img_cropped2.size:
        img_cropped2 = cv2.resize(img_cropped2,(img_cropped1.shape[1],img_cropped1.shape[0]))
    if img_cropped1.size < img_cropped2.size:
        img_cropped1 = cv2.resize(img_cropped1,(img_cropped2.shape[1],img_cropped2.shape[0]))    
    else:
        img_cropped2 = cv2.resize(img_cropped2,(img_cropped1.shape[1],img_cropped1.shape[0])) 
    img=cv2.hconcat([img_cropped1, img_cropped2])
    img_normalized = truncation_normalization(img)
    cl1 = clahe(img_normalized, 1.0)
    cl2 = clahe(img_normalized, 2.0)
    img_final = cv2.merge((np.array(img_normalized*255, dtype=np.uint8),cl1,cl2))
    # Resize the image to the final shape. 
    img_final = cv2.resize(img_final, (IMG_SIZE, IMG_SIZE))
    img_final=img_final/255
    return img_final
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# # GradCAM implementations and some utility tools
# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM, EigenGradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from mamo_preprocess import *
# import models_torch as models
# def weights_init_kaiming(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
#     elif classname.find('Linear') != -1:
#         nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
#     elif classname.find('BatchNorm') != -1:
#         # nn.init.uniform(m.weight.data, 1.0, 0.02)
#         m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
#         nn.init.constant(m.bias.data, 0.0)

path="Y:\\mamo\\TEKNOFEST_MG_EGITIM_1\\"
csv_file="veribilgisi_31_01_23.xlsx"
df=pd.read_excel(path+csv_file)
birads_0=list()
birads_1=list()
birads_4=list()
for aq in range(len(df)):
    if df.loc[aq]["BIRADS KATEGORİSİ"]=='BI-RADS0':
        birads_0.append(df.loc[aq])
    if df.loc[aq]["BIRADS KATEGORİSİ"]=='BI-RADS1-2':
        birads_1.append(df.loc[aq])
    if df.loc[aq]["BIRADS KATEGORİSİ"]=='BI-RADS4-5':
        birads_4.append(df.loc[aq])
df_final=list()
df_final.extend(birads_0[:200])
df_final.extend(birads_1[:200])
df_final.extend(birads_4[:200])
df_final = shuffle(df_final)
df_final=pd.DataFrame(df_final)
df_birads0=pd.DataFrame(birads_0)
df_birads1=pd.DataFrame(birads_1)
df_birads4=pd.DataFrame(birads_4)
conted_0=df_birads0["MEME KOMPOZİSYONU"].value_counts()
conted_1=df_birads1["MEME KOMPOZİSYONU"].value_counts()
conted_4=df_birads4["MEME KOMPOZİSYONU"].value_counts()
def sinif(listleq):
    listleq=listleq.reset_index()
    listleq.drop('index', inplace=True, axis=1)
    A_list=list()
    B_list=list()
    C_list=list()
    D_list=list()

    for a in range(len(listleq)):
        if listleq.loc[a]["MEME KOMPOZİSYONU"]=="A":
            A_list.append(listleq.loc[a])
        if listleq.loc[a]["MEME KOMPOZİSYONU"]=="B":
            B_list.append(listleq.loc[a])
        if listleq.loc[a]["MEME KOMPOZİSYONU"]=="C":
            C_list.append(listleq.loc[a])
        if listleq.loc[a]["MEME KOMPOZİSYONU"]=="D":
            D_list.append(listleq.loc[a])
    return A_list,B_list,C_list,D_list
def birlestirme(A,B,C,D):
    df_birads0_f=list()
    test_1=list()
    df_birads0_f.extend(A[:100])
    df_birads0_f.extend(B[:200])
    df_birads0_f.extend(C[:200])
    df_birads0_f.extend(D[:90])
    test_1.extend(A[100:])
    test_1.extend(B[200:])
    test_1.extend(C[200:])
    test_1.extend(D[90:])
    test_2=pd.DataFrame(test_1)
    test_2=test_2.reset_index()
    test_2.drop('index', inplace=True, axis=1)
    qrt=pd.DataFrame(df_birads0_f)
    qrt=qrt.reset_index()
    qrt.drop('index', inplace=True, axis=1)
    return qrt,test_2

A0,B0,C0,D0=sinif(df_birads0)

A1,B1,C1,D1=sinif(df_birads1)
A2,B2,C2,D2=sinif(df_birads4)

df_birads00,tested_0=birlestirme(A0,B0,C0,D0)
df_birads01,tested_1=birlestirme(A1,B1,C1,D1)
df_birads04,tested_4=birlestirme(A2,B2,C2,D2)
df_test=pd.concat([tested_0,tested_1,tested_4])
df_test = shuffle(df_test)

df_test=df_test.reset_index()
df_test.drop('index', inplace=True, axis=1)
df_final=pd.concat([df_birads00,df_birads01,df_birads04])
df_final = shuffle(df_final)

df_final=df_final.reset_index()
df_final.drop('index', inplace=True, axis=1)
class_idx={'BI-RADS0':0,'BI-RADS1-2':1,'BI-RADS4-5':2}
# class_idx={'BI-RADS0_A': 0, 'BI-RADS1-2_A': 1, 'BI-RADS4-5_A': 2,
#             'BI-RADS0_B': 3, 'BI-RADS1-2_B': 4, 'BI-RADS4-5_B': 5
#             ,'BI-RADS0_C': 6, 'BI-RADS1-2_C': 7, 'BI-RADS4-5_C': 8,
#             'BI-RADS0_D': 9, 'BI-RADS1-2_D': 10, 'BI-RADS4-5_D': 11}
idx2class={v: k for k, v in class_idx.items()}
varss=["LCC.dcm","LMLO.dcm","RCC.dcm","RMLO.dcm"]
def dict_prep(arr1,arguman):
                x_batch1=pydicom.dcmread(path+str(arguman["HASTANO"][arr1])+"\\" +varss[0])
                x_batch1=x_batch1.pixel_array
                x_batch3=pydicom.dcmread(path+str(arguman["HASTANO"][arr1])+"\\"+varss[2])
                x_batch3=x_batch3.pixel_array
                image1 = concat_preproces(x_batch3, x_batch1)
                image1 = image1.swapaxes(0,2).swapaxes(1,2)
                
                x_batch2=pydicom.dcmread(path+str(arguman["HASTANO"][arr1])+"\\"+varss[1])
                x_batch2=x_batch2.pixel_array
                x_batch4=pydicom.dcmread(path+str(arguman["HASTANO"][arr1])+"\\"+varss[3])
                x_batch4=x_batch4.pixel_array
                image2 = concat_preproces(x_batch4, x_batch2)
                image2 = image2.swapaxes(0,2).swapaxes(1,2)
                
                return image1,image2
cumulative_loss=list()
def TrainModelInBatchesV1(model, loss_func, optimizer, epochs=250):
    for epoch in range(epochs):
        losses = [] ## Record loss of each batch
        cumulative_loss.append(losses)
        for i in range(len(df_final)):
                hasta=list()
                x_batch1=pydicom.dcmread(path+str(df_final["HASTANO"][i])+"\\"+varss[0])
                x_batch1=x_batch1.pixel_array
                x_batch3=pydicom.dcmread(path+str(df_final["HASTANO"][i])+"\\"+varss[2])
                x_batch3=x_batch3.pixel_array
                image1 = concat_preproces(x_batch3, x_batch1)
                image1=torch.tensor(image1).type(torch.FloatTensor).cuda()
                image1 = image1.swapaxes(0,2).swapaxes(1,2).unsqueeze(0)
                hasta.append(image1)
                x_batch2=pydicom.dcmread(path+str(df_final["HASTANO"][i])+"\\"+varss[1])
                x_batch2=x_batch2.pixel_array
                x_batch4=pydicom.dcmread(path+str(df_final["HASTANO"][i])+"\\"+varss[3])
                x_batch4=x_batch4.pixel_array
                image2 = concat_preproces(x_batch4, x_batch2)
                image2=torch.tensor(image2).type(torch.FloatTensor).cuda()
                image2 = image2.swapaxes(0,2).swapaxes(1,2).unsqueeze(0)
                hasta.append(image2)
                if i%2==0:
                    y_batch=df_final["BIRADS KATEGORİSİ"][i]
                    # kompo=df_final["MEME KOMPOZİSYONU"][i]
                    y_batch=torch.tensor(class_idx[f"{y_batch}"]).cuda()       
                    y_batch=y_batch.unsqueeze(0)
                for k in range(2):
                    x=hasta[k]
                    preds = model(x) ## Make Predictions by forward pass through network
                    loss = loss_func(preds, y_batch) ## Calculate Loss

                    losses.append(loss.detach().cpu().numpy()) ## Record Loss

                    optimizer.zero_grad() ## Zero weights before calculating gradients
                    loss.backward() ## Calculate Gradients
                    optimizer.step() ## Update Weights
                
                if i%100==0:
                    print(f"Calisiyor kanka: {i} epoch: {epoch}")
                    print(f"Toplam loss: {np.mean(losses)}")
                print(f"{i}_{loss}")
                if np.mean(losses)==0.1 or np.mean(losses)==0.05 or np.mean(losses)==0.009:
                    torch.save(model.state_dict(), f"mamo_12_class_01_03_normalize_data{np.mean(losses)}.torch")
        scheduler.step()
        print("Train CategoricalCrossEntropy : {:.3f}".format(np.mean(losses)))
        
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device= torch.device('cpu')  # train on the GPU or on the CPU, if a GPU is not available
from torch.optim import SGD, RMSprop, Adam
path_model=r"D:\Akademik dosyalar\Yabay_Zeka(Ne_yabayi)\mamo_agirlik\mamo_02_03_densenet_0_005.torch"
def get_densenet121(pretrained=False, out_features=None, path=None):
    model = torchvision.models.densenet121(pretrained=pretrained)
    if out_features is not None:
        model.classifier = torch.nn.Linear(
            in_features=1024, out_features=out_features
        )
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)
def get_resnet18(pretrained=False, out_features=None, path=None):
    model = torchvision.models.resnet18(pretrained=pretrained)
    if out_features is not None:
        model.fc = torch.nn.Linear(in_features=512, out_features=out_features)
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)
model= get_densenet121(pretrained=True,out_features=(3),path=path_model).cuda()

epochs = 500
learning_rate = 0.0001 # 0.001

cross_entropy_loss = nn.CrossEntropyLoss()
cross_entropy_loss.cuda()
optimizer = Adam(params=model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
TrainModelInBatchesV1(model, cross_entropy_loss, optimizer,epochs)

# torch.save(model.state_dict(), r"D:\Akademik dosyalar\Yabay_Zeka(Ne_yabayi)\mamo_agirlik\mamo_02_03_densenet_0_005.torch")
# model.load_state_dict(torch.load(r"D:\Akademik dosyalar\Yabay_Zeka(Ne_yabayi)\mamo_agirlik\mamo_01_03_densenet_0_005.torch"))
test_loc=1500
x_test1,x_test2=dict_prep(test_loc,df_test)

x_test1=torch.tensor(x_test1).type(torch.FloatTensor).cpu().unsqueeze(0)
x_test2=torch.tensor(x_test2).type(torch.FloatTensor).cpu().unsqueeze(0)

pred1 = model(x_test1)
pred2 =model(x_test2)


import torch.nn.functional as F
F.softmax(pred1, dim=-1).argmax(), F.softmax(pred1, dim=-1).max()
print("Predicted Target : {}".format(idx2class[pred1.argmax(dim=-1).item()]))
print("GT:{}".format(df_test["BIRADS KATEGORİSİ"][test_loc]))
print("GT:{}".format(df_test["MEME KOMPOZİSYONU"][test_loc]))

F.softmax(pred2, dim=-1).argmax(), F.softmax(pred2, dim=-1).max()
pred_plus=pred1+pred2
print("Predicted Target : {}".format(idx2class[pred2.argmax(dim=-1).item()]))
print("Predicted Target_plus : {}".format(idx2class[pred_plus.argmax(dim=-1).item()]))
print("GT:{}".format(df_test["BIRADS KATEGORİSİ"][test_loc]))
print("GT:{}".format(df_test["MEME KOMPOZİSYONU"][test_loc]))
def data_check(listeler,values):
    gt_1=[]
    pred_1=[]
    for i in range(values):
        x_test1,x_test2=dict_prep(i,df_test)
        x_test1=torch.tensor(x_test1).type(torch.FloatTensor).cpu().unsqueeze(0)
        x_test2=torch.tensor(x_test2).type(torch.FloatTensor).cpu().unsqueeze(0)

        pred1 = model(x_test1)
        pred2 =model(x_test2)
        pred_plus=pred1+pred2


        gt_2="{}".format(listeler["BIRADS KATEGORİSİ"][i])
        pred_2=idx2class[pred_plus.argmax(dim=-1).item()]
        gt_1.append(gt_2)
        pred_1.append(pred_2)
    ground=pd.Series(gt_1,dtype="str",name="Ground_truth")
    predicted=pd.Series(pred_1,dtype="str",name="Predicted")
    return pd.concat([ground,predicted],axis=1)
kontrol=data_check(df_test,len(df_test))
dogrular=list()
for c in range(len(df_test)):
    if kontrol["Ground_truth"][c]==kontrol["Predicted"][c]:
        dogrular.append(c)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.featuremaps = []
        self.gradients = []

        target_layer.register_forward_hook(self.save_featuremaps)
        target_layer.register_backward_hook(self.save_gradients)

    def save_featuremaps(self, module, input, output):
        self.featuremaps.append(output)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def get_cam_weights(self, grads):
        return np.mean(grads, axis=(1, 2))

    def __call__(self, image, label=None):
        preds = self.model(image)
        self.model.zero_grad()

        if label is None:
            label = preds.argmax(dim=1).item()

        preds[:, label].backward()

        featuremaps = self.featuremaps[-1].cpu().data.numpy()[0, :]
        gradients = self.gradients[-1].cpu().data.numpy()[0, :]

        weights = self.get_cam_weights(gradients)
        cam = np.zeros(featuremaps.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * featuremaps[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, image.shape[-2:][::-1])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return label, cam
##cv2.COLORMAP_JET    
def apply_mask(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

label=class_idx['BI-RADS4-5_C']

target_layer = model.features[-1]

cam = GradCAM(model=model.float(), target_layer=target_layer)

label, mask1 = cam(x_test1.float(), label)
label, mask2 = cam(x_test2.float(), label)

img1,img2=dict_prep(test_loc,df_test)
img1=img1.swapaxes(1,2).swapaxes(0,2)
img2=img2.swapaxes(1,2).swapaxes(0,2)
cc = apply_mask(img1, mask1)
mlo = apply_mask(img2, mask2)
plt.figure(1)
plt.imshow(cc)
plt.figure(2)
plt.imshow(mlo)

# def data_check(listeler,values):
#     gt_1=[]
#     pred_1=[]
#     for i in range(values):
#         tested=dict_prep(i,df)
#         pred_3 = model(tested)

#         gt_2="{}_{}".format(listeler["BIRADS KATEGORİSİ"][i],listeler["MEME KOMPOZİSYONU"][i])
#         pred_2=idx2class[pred_3.argmax(dim=-1).item()]
#         gt_1.append(gt_2)
#         pred_1.append(pred_2)
#     ground=pd.Series(gt_1,dtype="str",name="Ground_truth")
#     predicted=pd.Series(pred_1,dtype="str",name="Predicted")
#     return pd.concat([ground,predicted],axis=1)
# kontrol=data_check(df_test,len(df_test))
# dogrular=list()
# for c in range(len(df_test)):
#     if kontrol["Ground_truth"][c][:-2]==kontrol["Predicted"][c][:-2]:
#         dogrular.append(c)
