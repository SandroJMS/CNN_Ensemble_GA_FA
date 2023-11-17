###############################################################
# Title: CNNs Ensemble with Firefly Algorithm and Genetic     #
# Algorithm                                                   #
# Author: Sandro Jose Mendes Sobrinho                         #
# Date: 6 Nov 2023                                            #
###############################################################

#For better results you can adjust the parameters of the ensemble algorithms

###############################################################
# Import libs                                                 #
###############################################################

import os
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
from numpy.random import default_rng, random
import pygad

###############################################################
# Data Loader                                                 #
###############################################################
def create_dataloader(main_folder):
    #Get folder paths
    train_dir = os.path.join(main_folder,"train")
    test_dir = os.path.join(main_folder,"test")
    
    #Input shape for models
    batch_size = 16
    image_size = (256, 256)
    
    #Create train, validation and test data loaders
    train_data_loader = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed = 10,
        label_mode='categorical',
        validation_split=0.1, 
        subset='training', 
    )
    validation_data_loader = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed = 10,
        label_mode='categorical',
        validation_split=0.1,
        subset='validation' 
    )
    test_data_loader = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=10,
        label_mode='categorical'
    )
    
    #Data augmentation
    aug1 = train_data_loader.map(lambda x, y: (augment(x), y))
    train_data_loader = train_data_loader.concatenate(aug1)
    
    #Normalization
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_data_loader = train_data_loader.map(lambda x, y: (normalization_layer(x), y))
    validation_data_loader = validation_data_loader.map(lambda x, y: (normalization_layer(x), y))
    test_data_loader = test_data_loader.map(lambda x, y: (normalization_layer(x), y))
    
    return train_data_loader,test_data_loader,validation_data_loader


###############################################################
# Data Augmentation                                           #
############################################################### 

def augment(image):
    #Set the image data as float 32
    image = tf.cast(image, tf.float32)
    
    #Changes the bright, contrast, hue and introduces a random chance of flipping the image
    image = tf.image.random_brightness(image, max_delta=.2)
    image = tf.image.random_contrast(image,.5,.95)
    image = tf.image.random_hue(image,.1)
    image = tf.image.random_flip_left_right(image)
    return image

###############################################################
# Utils                                                       #
###############################################################

#Get model predictions of the dataset
def get_models_outputs(model_list, val_imgs):
    preds = []
    for model in model_list:
        h5_model  = tf.keras.models.load_model(model+'.h5')
        pred = h5_model.predict(val_imgs)
        
        preds.append(pred)
    return np.stack(preds) 
    
#Binary cross-entropy function
def BCE_func(solution,prediction,validation,inversion=False):
    solution = NormMatrix(solution)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    shape = (prediction.shape[0],prediction.shape[-1])
    weight_arr = np.reshape(solution,(shape))
    mult_matrix = np.array([prediction[:,i,:]*weight_arr for i in range(prediction.shape[1])])
    mult_matrix = np.sum(mult_matrix,axis=1)
    mult_matrix = np.transpose(mult_matrix)
    mult_matrix = NormMatrix(mult_matrix)
    mult_matrix = np.transpose(mult_matrix)
    score = bce(validation, mult_matrix).numpy()
    if inversion:
        return (1/score)
    else:
        return score

#Accuracy function    
def acc_func(solution,prediction,validation):
    solution = NormMatrix(solution)
    shape = (prediction.shape[0],prediction.shape[-1])
    weight_arr = np.reshape(solution,(shape))
    mult_matrix = np.array([prediction[:,i,:]*weight_arr for i in range(prediction.shape[1])])
    mult_matrix = np.sum(mult_matrix,axis=1)
    mult_matrix = np.transpose(mult_matrix)
    mult_matrix = NormMatrix(mult_matrix)
    mult_matrix = np.transpose(mult_matrix)
    acc = accuracy_score(np.argmax(validation, axis = 1), np.argmax(mult_matrix, axis = 1))
    return acc

#Get metrics for each loaded model
def test_models(pred_val, label_val, pred_test, label_test):
    for model in range(pred_val.shape[0]):
        print("\nTesting Model - ", model)
        val_matrix = np.zeros((pred_val.shape[0],pred_val.shape[-1]))
        val_matrix[model,:]=1
        print("objective - ",BCE_func(val_matrix,pred_val,label_val))
        print("acc - ",acc_func(val_matrix,pred_val,label_val))
        print("test objective - ",BCE_func(val_matrix,pred_test,label_test))
        print("test acc - ",acc_func(val_matrix,pred_test,label_test))

#Normalization of the matrix (Used for weight matrix and result matrix)
def NormMatrix(matrix):
    nMatrix = np.divide(matrix , (np.sum(matrix,axis=0)))
    return nMatrix

def getLabelAndPrediction(validation_data_loader,test_data_loader, model_list):
    val_imgs = []
    val_label = []

    for x, y in validation_data_loader:
        val_imgs.append(x)
        val_label.append(y)
    val_imgs = np.concatenate(val_imgs)
    val_label = np.concatenate(val_label)

    test_imgs = []
    test_label = []

    for x, y in test_data_loader:
        test_imgs.append(x)
        test_label.append(y)
    test_imgs = np.concatenate(test_imgs)
    test_label = np.concatenate(test_label)
                
    #Generate predictions
    val_pred = get_models_outputs(model_list, val_imgs)
    test_pred = get_models_outputs(model_list, test_imgs)
    
    return val_label,test_label,val_pred,test_pred


def EnsembleAlgorithmSelection():
    return int(input("\n1 - Firefly Algorithm\n2 - Genetic Algorithm\n3 - Exit\nNumber: "))

def SearchMissingModels(main_folder,model_list):
    return (False in [(model+'.h5' in os.listdir(main_folder)) for model in model_list])


###############################################################
# Firefly Algorithm                                           #
###############################################################    
class FireflyAlgorithm:
    
    ###############################################################
    # Initial Setup                                               #
    ############################################################### 
    
    #Set initial parameters for the Firefly Algorithm
    def __init__(self, pop_size=10, alpha=1.0, betamin=1.0, gamma=0.01, seed=None):
        self.pop_size = pop_size
        self.alpha = alpha
        self.betamin = betamin
        self.gamma = gamma
        self.rng = default_rng(seed)
            
    #Set the dataset parameters       
    def set_problem_params(self, label_val,pred_val, label_test, pred_test):
        self.label_val  = label_val
        self.pred_val   = pred_val
        self.label_test = label_test
        self.pred_test  = pred_test
        self.fire_shape = (pred_val.shape[0],pred_val.shape[-1])
        self.parameter_size = (pred_val.shape[0]*pred_val.shape[-1])
        
    ###############################################################
    # Solution Test Functions                                     #
    ###############################################################  
    
    def valid_BCE(self, fireflies):
        return BCE_func(fireflies,self.pred_val,self.label_val)
    
    def test_BCE(self, fireflies):
        return BCE_func(fireflies,self.pred_test,self.label_test)
    
    def valid_acc(self, fireflies):
        return acc_func(fireflies,self.pred_val,self.label_val)
    
    def test_acc(self, fireflies):
        return acc_func(fireflies,self.pred_test,self.label_test)
    
    ###############################################################
    # Main Function                                               #
    ############################################################### 

    def run(self, lb, ub, epochs):
        test_models( self.pred_val, self.label_val, self.pred_test, self.label_test)
        
        fireflies = self.rng.uniform(lb, ub, (self.pop_size, self.parameter_size))
        intensity = np.apply_along_axis(self.valid_BCE, 1, fireflies)
        best = np.min(intensity)
        best_idx = np.argmin(intensity,axis = 0)
        

        evaluations = 0
        new_alpha = self.alpha
        search_range = ub - lb

        while evaluations < epochs:
            evaluations += 1
            new_alpha *= 0.97
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if intensity[i] >= intensity[j]:
                        r = np.sum(np.square(fireflies[i] - fireflies[j]), axis=-1)
                        beta = self.betamin * np.exp(-self.gamma * r)
                        steps = new_alpha * (self.rng.random(self.parameter_size) - 0.5) * search_range
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + steps
                        fireflies[i] = np.clip(fireflies[i], lb, ub)
                        intensity[i] = self.valid_BCE(fireflies[i])
                        if intensity[i]<best:
                            best = intensity[i]
                            best_idx = i
                        
            
            print("\nepoch -",evaluations)
            print("objective - ",best)
            print("acc - ",self.test_acc(fireflies[best_idx]))
            print("test objective - ",self.test_BCE(fireflies[best_idx]))
            print("test acc - ",self.valid_acc(fireflies[best_idx]))
            
        solution = np.reshape(fireflies[best_idx],(self.fire_shape))
        solution = NormMatrix(solution)
        return  solution
    
    
    
###############################################################
# Genetic Algorithm                                           #
###############################################################  
class GeneticAlgorithm:
    
    ###############################################################
    # Initial Setup                                               #
    ############################################################### 
    
    def set_problem_params(self, label_val,pred_val, label_test, pred_test):
        self.label_val  = label_val
        self.pred_val   = pred_val
        self.label_test = label_test
        self.pred_test  = pred_test
        self.input_shape = (pred_val.shape[0],pred_val.shape[-1])
        self.parameter_size = (pred_val.shape[0]*pred_val.shape[-1])
        
    def on_gen(self,ga_instance):
        print("\n Generation : ", ga_instance.generations_completed)
        print("Fitness of the best solution :", 1/(ga_instance.best_solution()[1]))
        
    def set_ga(self, num_generations=50, num_parents= 10, pop_size=100, mutation_percent_genes=.15):
        self.num_gen = num_generations
        self.pop_size = pop_size
        self.num_parents = num_parents;
        self.mutation_chance = mutation_percent_genes
        return
        
        
        
        
    ###############################################################
    # Solution Test Functions                                     #
    ###############################################################  
        
    def valid_BCE(self, solution):
        return BCE_func(solution,self.pred_val,self.label_val,inversion=True)
    
    def test_BCE(self, solution):
        return BCE_func(solution,self.pred_test,self.label_test,inversion=False)
    
    def valid_acc(self, solution):
        return acc_func(solution,self.pred_val,self.label_val)
    
    def test_acc(self, solution):
        return acc_func(solution,self.pred_test,self.label_test)
    
    ###############################################################
    # Selection Wheel                                             #
    ############################################################### 
    def makeWheel(self, fitness_pop):
        wheel = []
        total = np.sum(fitness_pop)
        total_sum = 0
        for i in range(fitness_pop.shape[0]):
            p = (fitness_pop[i])/total
            wheel.append((i,total_sum,total_sum+p))
            total_sum = total_sum+p
        return wheel
    
    def selectN(self, wheel, N=-1):
        if N==-1:
            N = self.num_parents
        answer = []
        while len(answer) < N:
            r = random()
            for i,lb,ub in wheel:
                if (lb<=r and r<ub):
                    if i not in answer:
                        answer.append(i)
                    pass
        return answer
    
    ###############################################################
    # Main Function                                               #
    ############################################################### 
    
    def run(self, lb, ub, epochs):
        #generate initial population
        solutions = np.random.uniform(lb, ub, (self.pop_size, self.parameter_size))
        #calculate fitness
        fitness = np.apply_along_axis(self.valid_BCE, 1, solutions)
        best = np.argmax(fitness)
        
        for gen in range(self.num_gen):
       
            #Stochastic Universal Sampling Selection with elite selection
            wheel = self.makeWheel(fitness)
            selected = self.selectN(wheel)
            
            #if the best is not on the selection, manually remove the last one
            #and replace by the best value
            if best not in selected:
                selected = selected[:-1]
                selected.append(best)
            
            parents = solutions[selected]
            parents_fitness = fitness[selected]
            
            # Multi Point Crossover with random selected parents
            wheel_parents = self.makeWheel(parents_fitness)
            
            
            #Multi Point Crossover
            children = np.array([])
            for i in range(self.num_parents,self.pop_size):
                #Select Parents
                selected_parents = self.selectN(wheel_parents,2)
                parent1 = parents[selected_parents[0]]
                parent2 = parents[selected_parents[1]]
                #Select Crossover Points 
                two_points = [0,0]
                while (two_points[0]==two_points[1]):
                    two_points = np.sort(np.random.randint([0,0],[55,55]))
                    
                #Use crossover to create a new child
                child = np.concatenate((
                    parent1[0:two_points[0]],
                    parent2[two_points[0]:two_points[1]],
                    parent1[two_points[1]:]),axis=0)
                
                #Add the child to the children list
                child = np.reshape(child,(1,child.shape[0]))
                if children.shape[0]==0:     #first child
                    children = child
                else:                        
                    children = np.concatenate((children,child),axis=0)
                
            
            #Mutation
            for i in range(children.shape[0]): #for each child
                for j in range(children.shape[1]):#for each gene
                    if self.mutation_chance>random():#if mutations happens
                        #chance to double, reduce to half or anything in between
                        children[i,j] = children[i,j] * np.random.uniform(low=0.5, high=2)
                        if children[i,j]>1:
                            children[i,j]=1
            
            #Final population
            solutions = np.concatenate((parents,children),axis=0)
            fitness = np.apply_along_axis(self.valid_BCE, 1, solutions)
            best = np.argmax(fitness)
            #generation metrics
            print("\nepoch -",gen+1)
            print("objective - ",1/fitness[best])
            print("acc - ",self.test_acc(solutions[best]))
            print("test objective - ",self.test_BCE(solutions[best]))
            print("test acc - ",self.valid_acc(solutions[best]))
          
        solution = np.reshape(solutions[best],(self.input_shape))
        solution = NormMatrix(solution)
        return solution


###############################################################
# Main Program                                                #
###############################################################    
if __name__ == "__main__":   
    #Select the main folder (where the python script is located)
    main_folder= os.getcwd()
    #List with the available models
    model_list = ["MobileNetV2","DenseNet201","ResNet152V2","Xception","InceptionV3"]
    #Load the datasets into dataloaders
    train_data_loader,test_data_loader,validation_data_loader = create_dataloader(main_folder)
    #Search for missing models.
    if SearchMissingModels(main_folder,model_list):
        print("Not all models are saved.\nExiting.")
    else:
        #Get models predictions
        val_label,test_label,val_pred,test_pred = getLabelAndPrediction(validation_data_loader,test_data_loader,model_list)
        while True:
            try:
                #Selection menu
                ensemble_int = EnsembleAlgorithmSelection()
                
                #Firefly Algorithm
                if ensemble_int==1:
                    FA = FireflyAlgorithm(pop_size=55,      #population size
                                          alpha=.5,         #alpha
                                          betamin=.2,       #beta
                                          gamma=0.01,       #gamma
                                          seed=10)          #seed (for reproducibility)
                    
                    FA.set_problem_params(val_label, val_pred, test_label, test_pred)
                    
                    FA_result =FA.run(lb = 0, ub = 1, epochs = 40)
                    np.save('FA_result.npy', FA_result) # save
                    
               #Genetic Algorithm     
                elif ensemble_int==2:
                    GA = GeneticAlgorithm()
                    
                    GA.set_problem_params(val_label, val_pred, test_label, test_pred)
    
                    GA.set_ga(num_generations=100,
                              num_parents= 10, 
                              pop_size=100, 
                              mutation_percent_genes=.15)
                    
                    GA_result = GA.run(lb = 0, ub = 1, epochs = 40)
                    np.save('GA_result.npy', GA_result) # save
                    
                elif ensemble_int==3:
                    break
            except:
                print("Invalid Option")