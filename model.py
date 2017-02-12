import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
import utils
import tqdm

##################################################
#           Preprocess steps                     #
##################################################

# Constants definition 
# uncomment for different file name, also need to be changed in utils.py
#csv_file = 'driving_log.csv'
OFFSET = 0.25 # offset to be applied in left and right angles
batch = 64
# define the new size of images (width, height)
input_size=(32,16)

# Global containers fo data
X_train = []
y_train = []

# Get images from cvs file (driving_log.csv file must be 
# in a folder name "data" with images contained in a folder name "IMG")
data = utils.read_images_from_file()

print('Concatenating X_train,y_train...\n')

# execute loop of 782*64 = 50,048 random samples
for a in tqdm.tqdm(range(782)):
    
    # split angles and images
    angles,images = data    
    # random indexes for iteration, using len of center which is equal 
    # for the other images (left and right). 
    indexes = np.random.randint(0, len(images['center']), 64)
    
    for i in indexes:
        # random choice of center left or right
        camera_pos = np.random.choice(['center','left','right'])
        img = images[camera_pos][i]
        ang = angles[i]
        
        # adjust offet according to position
        if camera_pos == 'left':
            ang -= OFFSET
        elif camera_pos == 'right':
            ang += OFFSET
        # Execute transformations (rotation,shear,resize,brightness)   
        img,ang = utils.transform_image(img,ang,input_size=input_size)
        # addition to global container
        X_train.append(img)
        y_train.append(ang)

# reserve a small portion for test the trained model. (0.01%)
X_train,X_test, y_train, y_test = train_test_split(np.array(X_train),np.array(y_train),test_size=0.001, random_state=22)


########################################################
#               Model Architecture                     #
########################################################

# Definition of constants
drop_rate = 0.5
batch_size = 64
epochs = 100
input_shape=(16,32,1)

# addtion of dimension in order to feed the netwok
X_train = X_train[:,:,:,None]
X_test = X_test[:,:,:,None]

# Definition of layers for the model architecture
layers = [
            BatchNormalization(axis=1,input_shape=input_shape),
            Convolution2D(8, 5, 5, border_mode="same",activation='relu'),
            MaxPooling2D((2,2),(1,1),'same'),
            Convolution2D(16, 1, 1, border_mode='same', activation='relu'),
            MaxPooling2D((2,2),(1,1),'same'),
            Convolution2D(32, 3, 3, border_mode='same', activation='relu'),
            MaxPooling2D((2,2),(1,1),'same'),
            Convolution2D(64, 2, 2, border_mode='same', activation='relu'),
            MaxPooling2D((4,4),(4,4),'same'),
            Dropout(0.5),
            Flatten(),
            Dense(1)
        ]
model =  Sequential(layers)

# uncomment to see model summary 
#model.summary()

# Create model using optimizer Adam and loss function mean square error
model.compile(optimizer='adam',loss='mse')
# Checkpoint for saving the weights whenever validation loss improves
chkpnt = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only=True, monitor='val_loss')
# Stop training when validation loss fails to decrease after 3 epochs
stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

#############################################################
#                   Traninig Process                        #
#############################################################
try:
    # Train model for a validation set of 20%.
    model.fit(X_train,y_train, nb_epoch=epochs,verbose=1,batch_size=batch_size,\
             shuffle=True,validation_split=0.2,\
            callbacks=[ chkpnt, stop])
    # Save model to file model.json
    utils.save_model(model)
    # Plot regression using testing set
    utils.plot_prediction(model,X_test,y_test)
    
except KeyboardInterrupt:
    # incase of interruption save model and plot regression
    utils.save_model(model)
    utils.plot_prediction(model,X_test,y_test)























