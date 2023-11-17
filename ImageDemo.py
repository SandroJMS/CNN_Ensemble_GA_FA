###############################################################
# Title: CNNs Ensemble with Firefly Algorithm and Genetic     #
# Algorithm                                                   #
# Author: Sandro Jose Mendes Sobrinho                         #
# Date: 6 Nov 2023                                            #
###############################################################

import matplotlib.pyplot as plt
# !pip install matplotlib==3.7.3 #######
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable as SM 
import cv2
import tensorflow as tf
import os

#Normalization of the matrix (Used for weight matrix and result matrix)
def NormMatrix(matrix, axis):
    nMatrix = np.divide(matrix , (np.sum(matrix,axis=axis)))
    return nMatrix

def PlotWeights(GA_T, FA_T):
    fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True, figsize=(10, 10))
    ax1.title.set_text("Genetic Algorithm Weights")
    sns.heatmap(GA_T, linewidth=0.5, ax=ax1,vmin=0, vmax=1, cmap = "inferno", cbar=False, annot=True, fmt=".0%")
    ax2.title.set_text("Firefly Algorithm Weights")
    sns.heatmap(FA_T, linewidth=0.5, ax=ax2,vmin=0, vmax=1, cmap = "inferno", cbar=False, annot=True, fmt=".0%")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(SM(norm=None, cmap="inferno"), cax=cbar_ax, format='%.3f')
    plt.show()
    

def LoadWeights():
    GA_T = np.load("GA_result.npy")
    FA_T = np.load("FA_result.npy")
    PlotWeights(GA_T, FA_T)
    return GA_T, FA_T

def loadModels():
    model_list = ["MobileNetV2","DenseNet201","ResNet152V2","Xception","InceptionV3"]
    h5_list = [tf.keras.models.load_model(model+'.h5') for model in model_list]
    return h5_list

def PreProcessImg(ImgPath):
    print("reading image")
    imagepath = ImgPath
    img = cv2.imread(imagepath)
    img = img = cv2.resize(img,(256,256))
    img = img.reshape((1,256,256,3))
    img = img/255
    return img

def predict(ImgPath, model_list, GA_T, FA_T):
    classes = ['bus', 'car', 'truck','motorcycle']
    preds=[]
    img = PreProcessImg(ImgPath)
    for model in h5_list:
        pred = model.predict(img)
        preds.append(pred)
      
    teste = np.concatenate(preds, axis=0)
    
    GA_mult_matrix = teste*GA_T
    GA_mult_matrix = np.sum(GA_mult_matrix,axis=0)
    GA_mult_matrix = NormMatrix(GA_mult_matrix,0)
    print("GA - ",classes[np.argmax(GA_mult_matrix)])
    
    FA_mult_matrix = teste*FA_T
    FA_mult_matrix = np.sum(FA_mult_matrix,axis=0)
    FA_mult_matrix = NormMatrix(FA_mult_matrix,0)
    print("FA - ",classes[np.argmax(FA_mult_matrix)])
    return

def imgMenu(model_list, GA_T, FA_T):
    while True:
        try:
            ImgStr = input("\nPut the relative image path or 0 to exit.\nExample: test\\Car\\Image_6.jpg\nPath: ")
            if ImgStr=='0':
                return
            ImgPath = os.getcwd()+'\\'+ImgStr
            predict(ImgPath, model_list, GA_T, FA_T)
        except:
            print("\nError Ocurred\nRestarting image menu...")
            pass


if __name__ == "__main__":
    #load generated weights for ensemble
    GA_T, FA_T = LoadWeights()
    #load CNN models
    print("\nloading models")
    h5_list = loadModels()
    #User input and predict
    imgMenu(h5_list, GA_T, FA_T)