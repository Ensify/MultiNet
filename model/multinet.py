import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, concatenate
from tensorflow.keras.models import Model

from model.timnet import TimNet
from model.pattlite import PAttLite

class MultiNet:
    def __init__(self, timnet_input_shape, class_labels, p_att_lite_input_shape, dropout_rate):
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.timnet_model = self.create_timnet_model(timnet_input_shape, class_labels)
        self.p_att_lite_model = self.create_p_att_lite_model(p_att_lite_input_shape, self.num_classes, dropout_rate)
        self.combined_model = self.create_combined_model(dropout_rate)
    
    def create_timnet_model(self, input_shape, class_labels):
        timnet = TimNet(input_shape=input_shape, class_label=class_labels)
        timnet.create_model(filter_size=39, kernel_size = 2, stack_size = 1, dilation_size = 8, dropout = 0.1)
        return timnet.model

    def create_p_att_lite_model(self, input_shape, num_classes, dropout_rate):
        inputs = Input(shape=input_shape)
        p_att_lite = PAttLite(input_shape=input_shape, num_classes=num_classes, train_dropout=dropout_rate, train_lr=0.001)
        outputs = p_att_lite(inputs)
        return Model(inputs, outputs)

    def create_combined_model(self, dropout_rate):
        # TIMNET part
        timnet_input = self.timnet_model.input
        timnet_output = self.timnet_model.layers[-2].output # Extracting the second last layer output
        # print(timnet_output)
        
        # PAttLite part
        p_att_lite_input = self.p_att_lite_model.input
        p_att_lite_output = self.p_att_lite_model.layers[-1].output # Extracting the second last layer output
        # print(p_att_lite_output)

        # Concatenate features
        combined_features = concatenate([timnet_output, p_att_lite_output])
        
        # Add classification head
        x = Dropout(dropout_rate)(combined_features)
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        final_output = Dense(self.num_classes, activation='softmax')(x)

        combined_model = Model(inputs=[timnet_input, p_att_lite_input], outputs=final_output)
        
        return combined_model

    def compile_model(self, learning_rate=0.001):
        self.combined_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=3.0),
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

    def train_model(self, timnet_data, p_att_lite_data, labels, batch_size, epochs, validation_data):
        return self.combined_model.fit([timnet_data, p_att_lite_data], labels,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=validation_data)

model = MultiNet(
    timnet_input_shape=(215,39), 
    class_labels=("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise"),
    p_att_lite_input_shape=(224, 224, 3), 
    dropout_rate=0.5
)
model.compile_model(learning_rate=0.001)
model.combined_model.summary()