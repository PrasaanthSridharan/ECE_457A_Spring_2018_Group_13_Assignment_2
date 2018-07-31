import tensorflow as tf
import numpy as np


class TensorFlowMNIST:

    def __init__(self, tf_data, hid_lay, hid_nod, hid_act, tf_ep=5,
                 tf_opt=tf.keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
                 tf_loss='sparse_categorical_crossentropy', tf_met=['accuracy']):
        """
        Initialize parameters for the tensor flow model and compile the model. This model uses 'sgd' as the optimizer,
        'sparse_categorical_crossentropy' for the loss function, and 'accuracy' as the metric

        :param tf_data: <String> The path to the MNIST data set
        :param hid_lay: <Int> The number of hidden layers
        :param hid_nod: <Array> Integer Array containing number of nodes for each hidden layer
        :param hid_act: <Array> tf.nn._ Array containing activation function for each hidden layer
        :param tf_ep: <Int> The number of epochs to train the model
        :param tf_opt: The optimization function to use when training model (For updating parameters)
        :param tf_loss: <String> The loss function to use when training model
        :param tf_met: <Array> String Array containing the metrics to use when training
        """

        # Loading test and training data from data_path
        self.x_train, self.y_train, self.x_test, self.y_test = TensorFlowMNIST.__load_data(tf_data)

        self.hid_lay = hid_lay
        self.hid_nod = hid_nod
        self.hid_act = hid_act
        self.tf_ep = tf_ep
        self.tf_opt = tf_opt
        self.tf_loss = tf_loss
        self.tf_met = tf_met

        self.model = None   # Initialize tf model variable

        seq_hid_layers = []
        for layer in range(self.hid_lay):
            seq_hid_layers.append(tf.keras.layers.Dense(self.hid_nod[layer], self.hid_act[layer]))

        self.model = tf.keras.models.Sequential(
            [tf.keras.layers.Flatten()] +
            seq_hid_layers +
            [tf.keras.layers.Dense(10, activation=tf.nn.softmax)]
        )

        self.model.compile(optimizer=self.tf_opt, loss=self.tf_loss, metrics=self.tf_met)

    def train(self, verbose=2):
        """
        Train and evaluate the tensor flow model

        :return: Get the evaluated model loss and accuracy
        """

        self.model.fit(self.x_train, self.y_train, epochs=self.tf_ep, verbose=verbose)
        return self.model.evaluate(self.x_test, self.y_test, verbose=verbose)

    @staticmethod
    def __load_data(tf_data):
        """
        Load the x_train, y_train and x_test, y_test variables from specified training and test data file

        :param tf_data: <String> Path containing training and test data file
        :return: training and test variables
        """

        f = np.load(tf_data)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        x_train, x_test = x_train / 255.0, x_test / 255.0
        f.close()

        return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    """
    THIS IS TEST CODE, achieves ~93% test accuracy.

    For applying the metaheuristics, you should consider the following parameters:
        1. # of Hidden Layers
        2. # of nodes per hidden layer
        3. # of epochs to train (too much will cause over training, too little will result in not enough training)
        4. 'lr' value (this is the learning rate for the stochastic gradient decent)
        5. 'momentum' value (this is the momentum for stochastic gradient decent)
        6. 'decay' value (this is how much decay will occur during each step for stochastic gradient decent)
    """

    data_path = "../MNIST_data/mnist.npz"
    hid_lay = 1
    hid_node = [512]
    hid_act = [tf.nn.relu]
    tf_ep = 3
    tf_opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

    tf_model = TensorFlowMNIST(data_path, hid_lay, hid_node, hid_act, tf_ep)
    loss, accuracy = tf_model.train(2)

    print("Loss:", loss)
    print("Accuracy:", accuracy)
