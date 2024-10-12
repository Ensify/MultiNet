import tensorflow as tf
from tensorflow import keras

class PAttLite(tf.keras.Model):
    def __init__(self, input_shape, num_classes, train_dropout, train_lr):
        super(PAttLite, self).__init__()
        self.num_classes = num_classes
        self.train_dropout = train_dropout

        self.sample_resizing = tf.keras.layers.experimental.preprocessing.Resizing(224, 224, name="resize")
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(mode='horizontal'),
            tf.keras.layers.RandomContrast(factor=0.3)
        ], name="augmentation")
        self.preprocess_input = tf.keras.applications.mobilenet.preprocess_input

        self.backbone = tf.keras.applications.mobilenet.MobileNet(
            input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        self.backbone.trainable = False
        self.base_model = tf.keras.Model(self.backbone.input, self.backbone.layers[-29].output, name='base_model')

        self.self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')
        self.patch_extraction = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'),
            tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'),
            tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu')
        ], name='patch_extraction')
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        self.pre_classification = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization()
        ], name='pre_classification')
        self.prediction_layer = tf.keras.layers.Dense(64, name='feature_extraction')

    def call(self, inputs, training=False):
        x = self.sample_resizing(inputs)
        x = self.data_augmentation(x)
        x = self.preprocess_input(x)
        x = self.base_model(x, training=False)
        x = self.patch_extraction(x)
        x = self.global_average_layer(x)
        x = tf.keras.layers.Dropout(self.train_dropout)(x, training=training)
        x = self.pre_classification(x)
        x = self.self_attention([x, x])
        outputs = self.prediction_layer(x)
        return outputs