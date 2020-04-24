from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os, shutil

def create_dirs():
    # This is the path where you want to create these folders
    original_path = '.'
    
    # Create 3 folders train, validation and test
    train_dir = os.path.join(original_path,'train1')
    os.makedirs(train_dir, exist_ok=True)

    validation_dir = os.path.join(original_path,'validation1')
    os.makedirs(validation_dir, exist_ok=True)

    test_dir = os.path.join(original_path,'test1')
    os.makedirs(test_dir, exist_ok=True)

    # Create 2 sub-folders
    train_masked_dir = os.path.join(train_dir,'masked-gen')
    os.makedirs(train_masked_dir, exist_ok=True)

    train_unmasked_dir = os.path.join(train_dir,'unmasked-gen')
    os.makedirs(train_unmasked_dir, exist_ok=True)

    # Validation directory for cats and dogs
    validation_masked_dir = os.path.join(validation_dir,'masked-gen')
    os.makedirs(validation_masked_dir, exist_ok=True)
    validation_unmasked_dir = os.path.join(validation_dir,'unmasked-gen')
    os.makedirs(validation_unmasked_dir, exist_ok=True)

    # test directory for cats and dogs
    test_masked_dir = os.path.join(test_dir,'masked-gen')
    os.makedirs(test_masked_dir, exist_ok=True)
    test_unmasked_dir = os.path.join(test_dir,'unmasked-gen')
    os.makedirs(test_unmasked_dir, exist_ok=True)

    return (train_dir, test_dir, validation_dir)

def create_model():
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3))) # 148, 148, 32
    model.add(Dropout(0.2))
    model.add(MaxPool2D((2,2))) 
    model.add(Conv2D(64,(3,3),activation='relu')) 
    model.add(Dropout(0.2))
    model.add(MaxPool2D((2,2))) 
    model.add(Conv2D(128,(3,3),activation='relu')) 
    model.add(Dropout(0.2))
    model.add(MaxPool2D((2,2))) 
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(MaxPool2D((2,2))) 
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer = 'Adam',metrics=['accuracy'])
    return model

def load_and_normalize_dataset(dir, batch_size=25, class_mode='binary'):
    # Normalizing the image values between 0 and 1
    datagen = ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_directory(dir,target_size=(150,150),batch_size=batch_size,class_mode=class_mode, interpolation='bilinear')
    
def train(model, train_generator, validation_generator, epochs=20, visualize=True):
    filepath = './best_weights.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    current_time = datetime.now()
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[checkpoint])
    print("Elapsed Time", str(datetime.now() - current_time))
    if visualize:
        visualize_loss_and_acc(history, epochs)
    return history

def visualize_loss_and_acc(history, epochs):
    train_acc = history.history['accuracy']
    validation_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    epochs = range(1, epochs+1)
    plt.plot(epochs,train_acc,'bo',label='Training Accuracy')
    plt.plot(epochs,validation_acc,'b',label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs,train_loss,'bo',label='Training Loss')
    plt.plot(epochs,validation_loss,'b',label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
