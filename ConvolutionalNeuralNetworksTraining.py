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
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import logging

###############################################################
# Disable some TensorFlow notifications                       #
###############################################################
logging.getLogger('tensorflow').setLevel(logging.INFO)
tf.autograph.set_verbosity(level=0, alsologtostdout=False)

###############################################################
# Data Loaders                                                #
###############################################################
def create_dataloaders(main_folder):
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
# Convolutional Neural Networks                               #
###############################################################
def ModelLoader(model_name):
    #Import model without the last layer
    if model_name == "DenseNet201":
        model_dense = tf.keras.applications.DenseNet201(input_shape=(256,256,3),
                                                    weights = "imagenet",include_top=False)
    elif model_name =="MobileNetV2":
        model_dense = tf.keras.applications.MobileNetV2(input_shape=(256,256,3),
                                                    weights = "imagenet",include_top=False)
    elif model_name =="ResNet152V2":
        model_dense = tf.keras.applications.ResNet152V2(input_shape=(256,256,3),
                                                    weights = "imagenet",include_top=False)
    elif model_name =="Xception":
        model_dense = tf.keras.applications.Xception(input_shape=(256,256,3),
                                                    weights = "imagenet",include_top=False)
    elif model_name =="InceptionV3":
        model_dense = tf.keras.applications.InceptionV3(input_shape=(256,256,3),
                                                    weights = "imagenet",include_top=False)
    else:
        print("ERROR - Model not implemented")
        return False, None
    
    #Insert new layers (Pooling and Fully Connected Layers)
    out = model_dense.output
    out = GlobalAveragePooling2D()(out)
    out = Dense(1024, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(512, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(256, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(32, activation='relu')(out)
    pred = Dense(4, activation='softmax')(out)
    
    #Keras complete model 
    model = Model(inputs=model_dense.input, outputs=pred)
    
    return True, model

###############################################################
# CNN Training                                                #
###############################################################
def train_model(train_data_loader,test_data_loader,validation_data_loader,model_list):
    
    #Model selection menu
    print("\nSelect a model for training")
    for index in range(len(model_list)):
        print(index+1," - ",model_list[index])
    print(len(model_list)+1," - Exit (dont train)")
        
    #User input
    while True:
        try:
            model_selection = int(input("Input the model number: "))
            if 0<model_selection<=len(model_list):
                break
            elif model_selection==len(model_list)+1:
                return None
        except:
            pass
        
    #Model selection
    train_bool,model = ModelLoader(model_list[model_selection-1])
    
    #Model training
    if train_bool:
        #Set learning rate, loss and metric
        model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy'])
        
        # Define the early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',         # Monitor validation loss
            patience=4,                 # Number of epochs with no improvement after which training will stop
            verbose=1,                  # Display messages about the early stopping process
            restore_best_weights=True   # Restore model weights from the epoch with the best value of the monitored metric
        )
        
        #Training function
        model.fit(
            train_data_loader,
            validation_data=validation_data_loader,
            epochs = 20,
            callbacks=[early_stopping])
        
        #Save model
        model_save_name = model_list[model_selection-1]+'.h5'
        model.save(model_save_name)
        return model
    
    else:
        return None
    

if __name__ == "__main__":
    #Select the main folder (where the python script is located)
    main_folder= os.getcwd()
    #List with the available models
    model_list = ["MobileNetV2","DenseNet201","ResNet152V2","Xception","InceptionV3"]
    #Load the datasets into dataloaders
    train_data_loader,test_data_loader,validation_data_loader = create_dataloaders(main_folder)
    #Train a model
    train_model(train_data_loader,test_data_loader,validation_data_loader, model_list)