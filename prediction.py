from __future__ import print_function
from __future__ import absolute_import

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import inception_resnet_v2
import pandas as pd
import bson
import struct
from tqdm import *

import warnings
import os

import explore_data as Data

def main():
    data_dir = 'data'
    test_bson_path = os.path.join(data_dir, "test.bson")
    num_test_products = 1768182
    file_weight = 'model_kaggle.h5'
    model = inception_resnet_v2.InceptionResNetV2(include_top=True,weights=None,input_tensor=None,input_shape=None,pooling=None,bottleneck=None,classes=nb_class)
    model.load_weights(file_weight)


    submission_df = pd.read_csv(data_dir + "sample_submission.csv")
    submission_df.head()

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    data = bson.decode_file_iter(open(test_bson_path, "rb"))

    with tqdm(total=num_test_products) as pbar:
        for c, d in enumerate(data):
            product_id = d["_id"]
            num_imgs = len(d["imgs"])

            batch_x = np.zeros((num_imgs, 180, 180, 3), dtype=K.floatx())

            for i in range(num_imgs):
                bson_img = d["imgs"][i]["picture"]

                # Load and preprocess the image.
                img = load_img(io.BytesIO(bson_img), target_size=(180, 180))
                x = img_to_array(img)
                x = test_datagen.random_transform(x)
                x = test_datagen.standardize(x)

                # Add the image to the batch.
                batch_x[i] = x

            prediction = model.predict(batch_x, batch_size=num_imgs)
            avg_pred = prediction.mean(axis=0)
            cat_idx = np.argmax(avg_pred)

            submission_df.iloc[c]["category_id"] = Data.idx2cat[cat_idx]        
            pbar.update()

    submission_df.to_csv("my_submission.csv.gz", compression="gzip", index=False)