import yaml
import argparse
import os
from multinet.data import utils
from multinet.model.multinet import MultiNet
import tensorflow as tf

def load_config(config_path):
    """Load training configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path):
    # Load configuration
    config = load_config(config_path)

    # Load dataset
    train_images, train_audio_features, train_labels = utils.load_cached_dataset(config['train_data_cache_path'])
    test_images, test_audio_features, test_labels = utils.load_cached_dataset(config['test_data_cache_path'])

    # Initialize the model
    model = MultiNet(
        timnet_input_shape=(215, 39),
        class_labels=config['classes'],
        p_att_lite_input_shape=(224, 224, 3),
        dropout_rate=config['dropout_rate']
    )

    # Compile the model
    model.compile_model(learning_rate=config['learning_rate'])

    # Define callbacks for saving the model and other custom callbacks
    callbacks = []
    if config['keep_best']:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(config['save_dir'], 'best_model.h5'),
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            )
        )
    if config['save_weights_frequency'] > 0:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(config['save_dir'], 'model_epoch_{epoch:02d}.h5'),
                save_freq=config['save_weights_frequency'] * len(train_labels) // config['batch_size'],
                save_weights_only=True
            )
        )

    # Train the model
    history = model.train_model(
        timnet_data=train_audio_features,
        p_att_lite_data=train_images,
        labels=train_labels,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=([test_audio_features, test_images], test_labels),
        callbacks=callbacks
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a MultiNet model using YAML configuration.")
    parser.add_argument('--config', required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    main(args.config)
