import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers, optimizers, losses, constraints
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Dense

# run on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class WeightModel:
    def __init__(self, sequence_length, alpha=1.0, learning_rate=0.001, use_tf=True, constraint=None):
        if use_tf:
            self._model = TFWeightModel(sequence_length=sequence_length, alpha=alpha,
                                        learning_rate=learning_rate, constraint=constraint)
        else:
            self._model = NPWeightModel(sequence_length=sequence_length, alpha=alpha,
                                        learning_rate=learning_rate, constraint=constraint)

    def train(self, relevant, irrelevant, epochs=1000):
        self._model.train(relevant=relevant, irrelevant=irrelevant, epochs=epochs)

    def get_weights(self, norm=None):
        weights = self._model.get_weights()
        if norm == 'min-max':
            # noinspection PyArgumentList
            weights = (weights - weights.min()) / (weights.max() - weights.min())
        elif norm == 'non-neg':
            weights[weights < 0] = 0  # non-neg constraints
        return weights

########################################################################################################################


class MinMaxConstraint(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be limited 0 and 1."""

    def __call__(self, weights):
        return (weights - tf.reduce_min(weights)) / (tf.reduce_max(weights) - tf.reduce_min(weights))


class TFWeightModel:
    def __init__(self, sequence_length, alpha=1.0, learning_rate=0.001, constraint=None):

        self.alpha = alpha
        self.learning_rate = learning_rate

        # Input
        relevant_input = Input(shape=(sequence_length,), name="relevant_passage_input")
        irrelevant_input = Input(shape=(sequence_length,), name="irrelevant_passage_input")

        # Weights constraints
        self.constraint = None
        if constraint == 'non-neg':
            self.constraint = constraints.NonNeg()
        elif constraint == 'min-max':
            self.constraint = MinMaxConstraint()

        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        self.weight_layer = Dense(1, use_bias=False,
                                  kernel_initializer=initializers.RandomNormal(mean=0.5, stddev=0.05),
                                  kernel_constraint=self.constraint)

        # Loss
        relevant_score = self.weight_layer(relevant_input)
        irrelevant_score = self.weight_layer(irrelevant_input)
        loss_layer = tf.maximum(irrelevant_score - relevant_score + self.alpha, 0)

        self.pair_model = Model(inputs=[relevant_input, irrelevant_input], outputs=loss_layer)
        self.pair_model.compile(optimizer=self.optimizer,
                                loss=losses.mean_squared_error)

    def train(self, relevant, irrelevant, epochs=1000):
        self.pair_model.fit(x=[relevant, irrelevant], y=np.zeros(shape=(len(relevant))),
                            epochs=epochs, batch_size=1024, verbose=0)

    def get_weights(self):
        return self.weight_layer.weights[0].numpy().squeeze()


########################################################################################################################


class NPWeightModel:
    def __init__(self, sequence_length, alpha=1.0, learning_rate=0.001, constraint=None):
        self.alpha = alpha
        self.learning_rate = learning_rate

        self.constraint = constraint
        self.optimizer = AdamOptimizer(learning_rate=learning_rate)
        self.weights = np.random.normal(loc=0.5, scale=0.05, size=sequence_length)

    def train(self, relevant, irrelevant, epochs=1000):

        for i in range(epochs):
            # Boosted scores
            relevant_score = np.dot(relevant, self.weights)
            irrelevant_score = np.dot(irrelevant, self.weights)

            # Compute loss
            loss = np.maximum(irrelevant_score - relevant_score + self.alpha, 0)

            # weight_gradient = np.sign(np.minimum(self.weights * irrelevant, 0) * irrelevant)

            # Update weights
            gradient = np.multiply(loss, (irrelevant - relevant).T).T
            # self.weights = self.weights - self.learning_rate * gradient.mean(axis=0)
            self.weights = self.optimizer.update(iteration=i, weights=self.weights, gradient=gradient.mean(axis=0))

            # Weights constraints
            if self.constraint == 'non-neg':
                self.weights[self.weights < 0] = 0  # non-neg constraints
            elif self.constraint == 'min-max':
                self.weights = (self.weights - self.weights.min()) / (self.weights.max() - self.weights.min())

    def get_weights(self):
        return self.weights


class AdamOptimizer:

    """
    Adam: A Method for Stochastic Optimization
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m_moment_vector = 0
        self.v_moment_vector = 0

    def update(self, iteration, weights, gradient):
        iteration += 1

        self.m_moment_vector = self.beta_1 * self.m_moment_vector + (1 - self.beta_1) * gradient
        self.v_moment_vector = self.beta_2 * self.v_moment_vector + (1 - self.beta_2) * gradient ** 2

        m_bias_corrected = self.m_moment_vector / (1 - self.beta_1 ** iteration)
        v_bias_corrected = self.v_moment_vector / (1 - self.beta_2 ** iteration)

        return weights - self.learning_rate * (m_bias_corrected / (np.sqrt(v_bias_corrected) + self.epsilon))
