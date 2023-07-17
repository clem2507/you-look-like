import wandb
from wandb.keras import WandbCallback

import os
from pathlib import Path
import argparse
import tensorflow as tf

from datetime import datetime
from keras.models import load_model


def data_loader(batch):
    """
    Method used to load the images into a trained and a validation set

    Attributes:

    -batch: batch size to prepare the data for training

    Returns:

    -train_ds: training dataset with images
    -val_ds: validation dataset with images
    """

    img_height = 256
    img_width = 256
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'data/cnn/download_cropped',
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        'data/cnn/download_cropped',
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch)

    train_ds.class_names = [train_ds.class_names[i][:-5] for i in range(len(train_ds.class_names))]
    val_ds.class_names = [val_ds.class_names[i][:-5] for i in range(len(val_ds.class_names))]

    return train_ds, val_ds

def train(batch, epochs):
    '''
    Function to train the CNN model

    Parameters:

    -batch: batch size for training
    -epochs: number of epochs to train the model
    '''

    wandb.init(project="you-look-like", entity="clem2507")

    train_ds, val_ds = data_loader(batch=batch)

    num_classes = len(train_ds.class_names)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2)
    ])

    model = tf.keras.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),
        # 256
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        # 128
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        # 64
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        # 32
        tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        # 16
        tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        # 8

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    model.build((None, 256, 256, 3))

    print(model.summary())

    log_time = datetime.now().strftime('%d-%m-%Y %H-%M-%S')
    save_dir = f'weights/tf-cnn/{log_time}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{save_dir}/best.h5',
        monitor='val_loss',
        mode='max',
        save_best_only=True
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        restore_best_weights=True,
        min_delta=1e-3
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[model_checkpoint, early_stop, WandbCallback()]
    )


def predict(img):
    '''
    Method used to predict an input image with the CNN model weights

    Parameters:

    -img: input user face image

    Returns:

    -prediction: string with the predicted celebrity
    '''

    most_recent_weights = sorted(Path('weights/tf-cnn').iterdir(), key=os.path.getmtime)[::-1]
    most_recent_weights = [name for name in most_recent_weights if not (str(name).split('/')[-1]).startswith('.')]
    model_path = f'{str(most_recent_weights[0])}/best.h5'
    if os.path.exists(model_path):
        model = load_model(model_path, compile=True)
    else:
        raise Exception('model path does not exist')

    prediction = model.predict(img)
    return prediction


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train the model on')
    return parser.parse_args()


def main(p):
    train(**vars(p))


if __name__ == '__main__':
    # statement to avoid useless warnings during training
    # export TF_CPP_MIN_LOG_LEVEL=3
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Test cmd line
    # python cnn/tf_cnn.py --batch 32 --epochs 20

    opt = parse_opt()
    main(p=opt)