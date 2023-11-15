from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import tensorflow 
from keras.preprocessing.image import load_img
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from zipfile import ZipFile


def extract_features(images):
    features = []

    for image in tqdm(images):
        img = load_img(image, color_mode="grayscale")
        img = img.resize((128, 128))
        img = np.array(img)
        features.append(img)

    features = np.array(features)

    features = features.reshape(len(features), 128, 128, 1)
    return features

def AgePredictionView(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.save()

            # Define the image path here
            image_path = os.path.join(settings.MEDIA_ROOT, str(image.image))

            zip_file_path = 'utk-face.zip'
            extraction_directory = 'extracted_images'
            os.makedirs(extraction_directory, exist_ok=True)
            with ZipFile(zip_file_path, 'r') as zip:
                zip.extractall(extraction_directory)
            utk_face_directory = os.path.join(extraction_directory, 'utk-face')
            image_paths = []
            age_labels = []
            gender_labels = []

            for filename in os.listdir(utk_face_directory):
                image_path = os.path.join(utk_face_directory, filename)
                temp = filename.split('_')
                age = int(temp[0])
                gender = int(temp[1])
                image_paths.append(image_path)
                age_labels.append(age)
                gender_labels.append(gender)

            df = pd.DataFrame()
            df['Image'], df['Age'], df['Gender'] = image_paths, age_labels, gender_labels
            X = extract_features(df['Image'])
            X = X / 255.0
            y_gender = np.array(df['Gender'])
            y_age = np.array(df['Age'])
            input_shape = (128, 128, 1)
            inputs = Input(input_shape)
            conv_1 = Conv2D(32, kernel_size=(3,3), activation = 'relu') (inputs)
            maxp_1 = MaxPooling2D(pool_size=(2,2)) (conv_1)
            conv_2 = Conv2D(64, kernel_size=(3,3), activation = 'relu') (maxp_1)
            maxp_2 = MaxPooling2D(pool_size=(2,2)) (conv_2)
            conv_3 = Conv2D(128, kernel_size=(3,3), activation = 'relu') (maxp_2)
            maxp_3 = MaxPooling2D(pool_size=(2,2)) (conv_3)
            conv_4 = Conv2D(256, kernel_size=(3,3), activation = 'relu') (maxp_3)
            maxp_4 = MaxPooling2D(pool_size=(2,2)) (conv_4)

            flatten = Flatten() (maxp_4)

            dense_1 = Dense(256, activation='relu') (flatten)
            dense_2 = Dense(256, activation='relu') (flatten)

            dropout_1 = Dropout(0.3) (dense_1)
            dropout_2 = Dropout(0.3) (dense_2)

            output_1 = Dense(1, activation = 'sigmoid', name='gender_out')(dropout_1)
            output_2 = Dense(1, activation = 'relu', name='age_out')(dropout_2)

            model = Model(inputs=[inputs], outputs=[output_1, output_2])
            model.compile(loss=['binary_crossentropy', 'mse'], optimizer='adam', metrics=['accuracy'])

            model.fit(x=X, y=[y_gender, y_age], batch_size = 32, epochs = 30, validation_split = 0.1)


            # Process the uploaded image
            upload_image = load_img(image_path, color_mode="grayscale", target_size=(128, 128))
            processed_image = np.array(upload_image)
            processed_image = processed_image / 255.0
            processed_image = processed_image.reshape(1, 128, 128, 1)

            # Make predictions
            predictions = model.predict(processed_image)
            predicted_gender = "Male" if predictions[0][0] < 0.5 else "Female"
            predicted_age = predictions[1][0]
            predicted_age = int(predicted_age)
            return render(request, 'age_app/result.html', {'image': image, 'predicted_age': predicted_age, 'predicted_gender': predicted_gender})
    else:
        form = ImageUploadForm()

    return render(request, 'age_app/upload.html', {'form': form})
