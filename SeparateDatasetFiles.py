###############################################################
# Title: CNNs Ensemble with Firefly Algorithm and Genetic     #
# Algorithm                                                   #
# Author: Sandro Jose Mendes Sobrinho                         #
# Date: 6 Nov 2023                                            #
###############################################################

###############################################################
# Import libs                                                 #
###############################################################
import os
import random
import shutil

###############################################################
# Dataset Separation                                          #
###############################################################
main_folder=os.getcwd()
dataset_folder = os.path.join(main_folder,"dataset")
classes = list(os.listdir(dataset_folder))

sum = 0
for root, dirs, files in os.walk(dataset_folder):
    if files != []:
        print(root, "-->", len(files))
        sum += len(files)
        
trainstr = os.path.join(main_folder,"train")
teststr = os.path.join(main_folder,"test")

try:    
    shutil.rmtree(trainstr, ignore_errors=True)
    shutil.rmtree(teststr, ignore_errors=True)
except:
    pass

os.mkdir(trainstr)
os.mkdir(teststr)        


for t in classes:
    origin = os.path.join(dataset_folder,t) 
    
    destination_train = os.path.join(trainstr,t) 
    destination_test = os.path.join(teststr,t) 
    
    os.mkdir(destination_train)
    os.mkdir(destination_test)
    
    for root, dirs, files in os.walk(origin):
        random.shuffle(files)
        train_list = files[:int((0.8)*len(files))]
        test_list = files[int((0.8)*len(files)):]
        for f in train_list:
            shutil.copy(os.path.join(origin,f),  os.path.join(destination_train,f))
        for f in test_list:
            shutil.copy(os.path.join(origin,f),  os.path.join(destination_test,f))
            

print("\nTRAIN")
sum = 0
for root, dirs, files in os.walk(trainstr):
    if files != []:
        print(root, "-->", len(files))
        sum += len(files)
        
print("\nTEST")        
sum = 0
for root, dirs, files in os.walk(teststr):
    if files != []:
        print(root, "-->", len(files))
        sum += len(files)