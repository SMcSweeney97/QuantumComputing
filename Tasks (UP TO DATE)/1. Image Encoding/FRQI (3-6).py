import numpy as np
import pandas as pd
import pennylane as qml
import tensorflow as tf
from matplotlib import pyplot as plt

tf.keras.backend.set_floatx('float64')

#FRQI NOT YET IMPLEMENTED, THIS IS TAKEN FROM AMPLITUDE ENCODING FILES

class Preprocessing():

    def __init__(self, dataset):
        (x_train, y_train), (x_test, y_test) = dataset

        x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

        print('Original Training examples: {}'.format(len(x_train)))
        print('Original Test examples: {}'.format(len(x_test)))

        self.x_train, self.y_train = self.filter_classes(x_train, y_train)
        self.x_test, self.y_test = self.filter_classes(x_test, y_test)

        #self.x_train.reshape(1, -1)
        #self.x_test.reshape(1, -1)

        print('Number of filtered training examples: {}'.format(len(x_train)))
        print('Number of filtered test examples: {}'.format(len(x_test)))

        self.plot(x_train[0, :, :, 0], '28x28 Training Example')

        #x_train = self.shrink(x_train)
        #x_test = self.shrink(x_test)

        #self.plot(x_train[0, :, :, 0], '9x9 Training Example')

        #self.x_train = x_train[:, 3, :, :].reshape((-1, 9))
        #self.x_test = x_test[:, 3, :, :].reshape((-1, 9))

        #self.plot(self.x_train[0, :].reshape((1, -1)), '9x1 Training Example', vmin = 0, vmax = 1)
  
    def filter_classes(self, x, y):
        keep = (y == 3) | (y == 6)
        x, y = x[keep], y[keep]
        y = y == 3
        return x, y
    
    def plot(self, image, title, vmin = None, vmax = None):
        plt.imshow(image, cmap = 'Greys')
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(title, fontsize = 20)
        plt.show()

    def shrink(self, matrix):
        new_data = tf.image.resize(matrix, (9, 9)).numpy()
        
        return new_data

class QuantumNeuralNetwork():
    def __init__(self, x_train, y_train, x_test, y_test, epochs):
        self.epochs = epochs
        model_1l = self.generate_model(x_train, layers = 1)
        model_1l.compile(
            loss = tf.keras.losses.Hinge(),
            optimizer = tf.keras.optimizers.Adam(),
            metrics = [self.accuracy])
        history_1l, results_1l, model_1l = self.train_model(model_1l, x_train, y_train, x_test, y_test)
        model_1l.save('model_1l.h5')
        history_1l_df = pd.DataFrame(history_1l.history)
        with open('history_1l.csv', mode = 'w') as f:
            history_1l_df.to_csv(f)
        print('\nModel 1 complete!\n')

        #model_2l = self.generate_model(x_train, layers = 2)
        #model_2l.compile(
        #    loss = tf.keras.losses.Hinge(),
        #    optimizer = tf.keras.optimizers.Adam(),
        #    metrics = [self.accuracy])
        #history_2l, results_2l, model_2l = self.train_model(model_2l, x_train, y_train, x_test, y_test)
        #model_2l.save('model_2l.h5')
        #history_2l_df = pd.DataFrame(history_2l.history)
        #with open('history_2l.csv', mode = 'w') as f:
        #    history_2l_df.to_csv(f)
        #print('\nModel 2 complete!\n')

        #model_3l = self.generate_model(x_train, layers = 3)
        #model_3l.compile(
        #    loss = tf.keras.losses.Hinge(),
        #    optimizer = tf.keras.optimizers.Adam(),
        #    metrics = [self.accuracy])
        #history_3l, results_3l, model_3l = self.train_model(model_3l, x_train, y_train, x_test, y_test)
        #model_3l.save('model_3l.h5')
        #history_3l_df = pd.DataFrame(history_3l.history)
        #with open('history_3l.csv', mode = 'w') as f:
        #    history_3l_df.to_csv(f)
        #print('\nModel 3 complete!\n')

    def generate_model(self, x_train, layers):
        n_qubits = 10
        n_layers = layers
        dev = qml.device('default.qubit', wires = n_qubits)

        @qml.qnode(dev, diff_method = 'adjoint')
        def qnode(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires = range(10), pad_with = 240, normalize = False, do_queue = True, id = None)

            for ii in range(n_qubits):
                qml.RY(np.pi * inputs[ii], wires = ii)


            for jj in range(n_layers):
                for ii in range(n_qubits - 1):
                    qml.RZ(weights[jj, 2 * ii, 0], wires = 0)
                    qml.RY(weights[jj, 2 * ii, 1], wires = 0)
                    qml.RZ(weights[jj, 2 * ii, 2], wires = 0)

                    qml.RZ(weights[jj, 2 * ii + 1, 0], wires = ii + 1)
                    qml.RY(weights[jj, 2 * ii + 1, 1], wires = ii + 1)
                    qml.RZ(weights[jj, 2 * ii + 1, 2], wires = ii + 1)

                    qml.CNOT(wires = [ii + 1, 0])
                    
                qml.RZ(weights[jj, 2 * (n_qubits - 1), 0], wires = 0)
                qml.RY(weights[jj, 2 * (n_qubits - 1), 1], wires = 0)
                qml.RZ(weights[jj, 2 * (n_qubits - 1), 2], wires = 0)

            return qml.expval(qml.PauliZ(0))

        weight_shapes = {'weights': (n_layers, 2 * (n_qubits - 1) + 1, 3)}
        qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim = 1, name = 'quantumLayer')
        inputs = tf.keras.Input(shape = (1, 784), name = 'inputs')
        outputs = qlayer(inputs)
        model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'QNN')
        x = np.reshape(x_train[0,:], (1, 784))
        x = x[np.newaxis, :]
        model.predict(x)
        print(model.summary())
        
        return model

    def train_model(self, model, x_train, y_train, x_test, y_test):
        EPOCHS = self.epochs
        BATCH_SIZE = 32
        NUM_EXAMPLES = x_train.shape[0]

        x_train_sub = x_train[:NUM_EXAMPLES, :]

        y_train_hinge = 2.0 * y_train - 1.0
        y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]

        x_test_sub = x_test[:, :]
        y_test_sub = y_test[:]

        x_train_sub = np.reshape(x_train_sub[0,:], (1, 784))
        x_train_sub = x_train_sub[np.newaxis, :]

        x_test_sub = np.reshape(x_test_sub[0,:], (1, 784))
        x_test_sub = x_test_sub[np.newaxis, :]

        qnn_history = model.fit(
            x_train_sub,
            y_train_hinge_sub,
            batch_size = BATCH_SIZE,
            epochs = EPOCHS,
            verbose = 1,
            validation_data = (x_test_sub, y_test_sub))
        
        qnn_results = model.evaluate(x_test_sub, y_test_sub)
        
        return qnn_history, qnn_results, model

    def accuracy(self, y_true, y_pred):
        y_true = tf.squeeze(y_true) > 0.0
        y_pred = tf.squeeze(y_pred) > 0.0
        result = tf.cast(y_true == y_pred, tf.float32)

        return tf.reduce_mean(result)

class Plot():
    def __init__(self, history_1l, history_2l, history_3l, epochs):
        for col in history_1l.columns:
            plt.plot(np.arange(1, epochs + 1), history_1l[col], label = '1 layer QNN')
            plt.plot(np.arange(1, epochs + 1), history_2l[col], label = '2 layer QNN')
            plt.plot(np.arange(1, epochs + 1), history_3l[col], label = '3 layer QNN')
            plt.xlabel('Epochs')
            plt.ylabel(col)
            plt.legend()
            plt.xticks(np.arange(0, epochs + 1, 2))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.savefig('D:/OneDrive/Desktop/{}.png'.format(col), bbox_inches = 'tight', pad_inches = 0)
            plt.show()
    
def main():
    epochs = 20
    dataset = Preprocessing(tf.keras.datasets.mnist.load_data())
    x_train, y_train, x_test, y_test = dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test
    QuantumNeuralNetwork(x_train, y_train, x_test, y_test, epochs)

    history_1l = pd.read_csv('D:/OneDrive/Desktop/history_1l.csv')
    history_2l = pd.read_csv('D:/OneDrive/Desktop/history_2l.csv')
    history_3l = pd.read_csv('D:/OneDrive/Desktop/history_3l.csv')
    Plot(history_1l, history_2l, history_3l, epochs)

if __name__ == '__main__':
    main()