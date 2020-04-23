from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob, os

datagen = ImageDataGenerator(
                            rotation_range=40, 
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2, # Görüntünün bir kısmını kesiyor.
                            zoom_range=0.2,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='nearest',
                            )

filepaths = {
    "./train1/masked/*.jpg" : "./train1/masked-gen",
    "./test1/masked/*.jpg" : "./test1/masked-gen",
    "./train1/unmasked/*.jpg" : "./train1/unmasked-gen",
    "./test1/unmasked/*.jpg" : "./test1/unmasked-gen",
}

for source, target in filepaths.items():
    os.makedirs(target, exist_ok=True)
    print("Source", source, "Target", target)
    im_paths = glob.glob(source)
    imgs = []
    for path in im_paths:
        img = load_img(path)
        temp = img_to_array(img)
        temp = temp.reshape((1,)+temp.shape) # (1, temp.shape)
        imgs.append(temp)

    for index, img in enumerate(imgs):
        # Tek görüntüden 50 tane farklı görüntü üret ve .jpeg formatında ilgili klasöre kaydet.
        i = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir=target, save_format='jpg', save_prefix=f"{index}"):
            print("Image[{:0>4}/{:0>4}]".format(index,i))
            i+=1
            if i>50:
                break
    print("Source", source, "Target", target)
    print("Görüntüler kaydedildi.")