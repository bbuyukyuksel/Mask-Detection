import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from funcs import *
 
# Create Dirs
train_dir, test_dir, validation_dir = create_dirs()

# Visualize
train_masked_dir = os.path.join(train_dir, 'masked-gen')
fname = np.random.choice(os.listdir(train_masked_dir))
print("Filename", fname)
img = load_img(os.path.join(train_masked_dir, fname))
img = img.resize((150,150))
plt.imshow(img)
plt.show()

# Configs
epochs = 3
batch_size = 50

# Load Datasets
train_generator = load_and_normalize_dataset(train_dir, batch_size=batch_size)
test_generator = load_and_normalize_dataset(test_dir, batch_size=batch_size)

# Create Model
model = create_model()


# Train or Predict Flag
f_train = False

if f_train:
    history = train(model, train_generator, test_generator, epochs, visualize=True)
else:
    size = 4
    model.load_weights("00_best_weights.hdf5")
    #x=model.evaluate(test_generator)
    # One image predict
    #arr_img = img_to_array(img) / 255
    #arr_img = arr_img.reshape((1,)+arr_img.shape)
    #print("Class", model.predict_classes(arr_img))

    validation_generator = load_and_normalize_dataset(validation_dir, batch_size=(size**2), class_mode=None)
    images = next(validation_generator)
    predict_class = model.predict_classes(images)

    LABELS = [
        "Masked", "Unmasked",
    ]
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    for index, image in enumerate(images):
        if index == (size**2):
            break
        ax = plt.subplot(size,size, index+1)        
        plt.imshow(image)
        ax.set_title("{}".format( LABELS[predict_class[index][0]] ))
    plt.show()    
    

    