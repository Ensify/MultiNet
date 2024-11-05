import yaml
import argparse
import os
from multinet.data import utils  
from multinet.model.multinet import MultiNet  
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback


def load_config(config_path):
    """Load training configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def plot_live(history, save_dir):
    """Plot training accuracy and loss live."""
    epochs = range(1, len(history['loss']) + 1)

    # Clear the previous plot
    plt.clf()

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)  # Allows the plot to update live

    # Save the plot to the save directory
    plt.savefig(os.path.join(save_dir, 'training_plot.png'))

def live_plot_callback(save_dir):
    """Return a LambdaCallback to plot live training progress."""
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    
    def on_epoch_end(epoch, logs):
        # Append current epoch's logs to the history
        history['loss'].append(logs.get('loss'))
        history['val_loss'].append(logs.get('val_loss'))
        history['accuracy'].append(logs.get('accuracy'))
        history['val_accuracy'].append(logs.get('val_accuracy'))

        # Call the plot function with updated history
        plot_live(history, save_dir)
        
    return LambdaCallback(on_epoch_end=on_epoch_end)

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

    # Define callbacks
    callbacks = []

    # Plot live during training
    plt.ion()  # Interactive mode on for live updating plots

    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])
    callbacks.append(live_plot_callback(config['save_dir']))

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
        callbacks=callbacks  # Pass the callbacks list here
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a MultiNet model using YAML configuration.")
    parser.add_argument('--config', required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    main(args.config)