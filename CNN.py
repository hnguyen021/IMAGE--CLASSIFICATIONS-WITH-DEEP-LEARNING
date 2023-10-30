import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob
import imageio
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
def featureNormalization(X, X_mean, X_std):
    X_norm = (X - X_mean) / X_std
    return X_norm

def display_result(img, prob):
    plt.figure(facecolor="white")
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(img)

    plt.subplot(122)
    plt.barh([1], prob[0], 
        align='center', alpha=0.9)
    plt.yticks([1], ('Cat'))
    plt.xlim([0, 1])
    plt.show()
def predict_image(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = img_array/255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    

DATADIR_TRAIN = "Dataset/training_set"
DATADIR_TEST ="Dataset/test_set"
    
CATEGORIES = ["Dog", "Cat"]

IMG_SIZE = 50

training_data = []

def create_training_data(DATADIR,X,y):
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  #0 = chó, 1= mèo 

        for img in tqdm(os.listdir(path)):  # duyệt qua từng hình chó và mèo
            
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # chuyển thành mảng
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # Thay đổi kích thước để chuẩn hóa kích thước dữ liệu
            training_data.append([new_array, class_num])  # add this to our training_data            
    
    for features,label in training_data:
        X.append(features)
        y.append(label)
    return X,y
def built_model(X):
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  #đổi thuộc tính 3D thành 1D
    
    model.add(Dense(64))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
def main():
    X_train = []
    y_train = []
    X_test =[]
    
    y_test =[]
    X_train,y_train=create_training_data(DATADIR_TRAIN,X_train,y_train)
    #print(X_train)
    X_test,y_test=create_training_data(DATADIR_TEST,X_test,y_test)
    
    X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    #print(X_train)
    X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    
    #-----------training.
    
    X_train = X_train/255.0
    y_train = np.array(y_train)
    X_test = X_test/255.0
    y_test = np.array(y_test)
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    
    X_norm = featureNormalization(X_train, X_mean, X_std)
    X_train = X_norm
    X_norm = featureNormalization(X_test, X_mean, X_std)
    X_test = X_norm
    
    model = built_model(X_train)
    model.summary()
    history=model.fit(X_train, y_train, batch_size=32, epochs=25, validation_split=0.3)
    #history=model.fit(x=X_train, y=y_train, batch_size=None, epochs=8, verbose=1, callbacks=None, validation_split=0.3, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=903, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
    model.save('model_cnn.model')
    history1 =model.fit(X_test, y_test, batch_size=32, epochs=25, validation_split=0.3)
    #history1=model.fit(x=X_test, y=y_test, batch_size=None, epochs=8, verbose=1, callbacks=None, validation_split=0.3, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=1662, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
    print("accuracy in training data is",history.history['accuracy'][24])
    print("loss in training data is",history.history['loss'][24])
    print("accuracy in testing data is",history1.history['accuracy'][24])
    print("loss in testing data is",history1.history['loss'][24])
    # ve do thi cho trainng
    
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    # ve do thi cho testing
    plt.plot(history1.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history1.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['test'], loc='upper left')
    plt.show()
    #image = cv2.imread('./Dataset/dog_test2.jpg')
    dirs = './Dataset/checking'
    #dirs = r"/Dataset/checking"
    list_fn = glob.glob(dirs + "/*.jpg")
    
    
    model = tf.keras.models.load_model("model_cnn.model")
    
    prediction = model.predict([predict_image('Dataset/checking/cat3.jpg')])
    #print(prediction)  # will be a list in a list.
    
    print('model predict this is a:',CATEGORIES[int(prediction[0][0])])
    for fn in list_fn:
            img = imageio.imread(fn)       
            x=predict_image(fn)       
            x_norm = featureNormalization(x, X_mean, X_std)      
            prob = model.predict_proba(x_norm)
            print(prob)
            display_result(img, prob)
main()

