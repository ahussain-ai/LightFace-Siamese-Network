
import tensorflow as tf 
from tensorflow.keras import metrics
from tensorflow.keras.models import Model

class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0, name="contrastive_loss"):
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true, y_pred):
        # Ensure the same dtype for both inputs
        y_true = tf.cast(y_true, y_pred.dtype)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(self.margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

class SiameseModel(Model):
    """
    The Siamese Network model with custom training and testing loops.
    Computes the contrastive loss using the pairwise distances produced by the
    Siamese Network.
    """

    def __init__(self, siamese_network):
        super().__init__()
        self.siamese_network = siamese_network
        self.loss_tracker = metrics.Mean(name="loss")
        self.loss_fn = ContrastiveLoss()

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        (images1, images2), labels = data  # labels are 1 for similar, 0 for dissimilar pairs

        with tf.GradientTape() as tape:
            distances = self.siamese_network([images1, images2])
            loss = self.loss_fn(labels, distances)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))

        self.loss_tracker.update_state(loss)


        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        (images1, images2), labels = data

        distances = self.siamese_network([images1, images2])
        loss = self.loss_fn(labels, distances)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]
    

if __name__== '__main__' : 
    pass