
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
from tensorflow.keras.applications import resnet



# Define the base model
def get_base_model():

    # Define the target shape of the input images
    target_shape = (200, 200)
    base_cnn = resnet.ResNet50(
        weights="imagenet", input_shape=target_shape + (3,), include_top=False
    )

    flatten = Flatten()(base_cnn.output)
    dense1 = Dense(1024, activation="relu")(flatten)
    norm = BatchNormalization()(dense1)
    output = Dense(256, activation='sigmoid')(norm)

    embedding = Model(base_cnn.input, output, name="Embedding")

    # Freeze layers except for the top few
    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    return embedding

class EuclideanDistance(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.margin = 1e-04

    def call(self, inputs):
        featsA, featsB = inputs
        sum_squared = tf.reduce_sum(tf.square(featsA - featsB), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_squared, self.margin))

def make_siamese_model()  :

    input_a = Input(name="anchor", shape=target_shape + (3,))
    input_b = Input(name="positive", shape=target_shape + (3,))


    embedding = get_base_model()

    embeddings_a = embedding(resnet.preprocess_input(input_a))
    embeddings_b = embedding(resnet.preprocess_input(input_b))


    distance = EuclideanDistance()([embeddings_a, embeddings_b])
    siamese_network = Model(inputs=[input_a, input_b], outputs=distance)

    return siamese_network


if __name__ == '__main__' : 
    pass
