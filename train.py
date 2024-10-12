from multinet.model.multinet import MultiNet

model = MultiNet(
    timnet_input_shape=(215,39), 
    class_labels=("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise"),
    p_att_lite_input_shape=(224, 224, 3), 
    dropout_rate=0.5
)
model.compile_model(learning_rate=0.001)
model.combined_model.summary()