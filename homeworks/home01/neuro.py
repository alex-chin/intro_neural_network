import numpy as np


class Neuro_1:
    # массив для ошибок, чтобы потом построить график
    errors = []
    # массив весов
    w=[]
    # массив слоев
    layer=[]
    accuracy = 0

    def __init__(self, neuron_numb=3, learning_rate=0.05, num_epochs=10000):
        # neuron_numb определим число нейронов скрытого слоя
        self.neuron_numb = neuron_numb
        # скорость обучения (learning rate)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self._generate_weights()

    def _generate_weights(self):
        self.w[0] = 2 * np.random.random((4, self.neuron_numb)) - 1  # для входного слоя   - 4 входа, 3 выхода
        self.w[1] = 2 * np.random.random((self.neuron_numb, 3)) - 1  # для внутреннего слоя - 5 входов, 3 выхода

    def fit(self, X, y):
        layer0 = X

        # процесс обучения
        for i in range(self.num_epochs):
            # прямое распространение(feed forward)
            layer1 = self.sigmoid(np.dot(layer0, self.w[0]))
            layer2 = self.sigmoid(np.dot(layer1, self.w[1]))

            # обратное распространение(back propagation) с использованием градиентного спуска
            layer2_error = y - layer2  # производная функции потерь = производная квадратичных потерь
            layer2_delta = layer2_error * self.sigmoid_deriv(layer2)

            layer1_error = layer2_delta.dot(self.w[1].T)
            layer1_delta = layer1_error * self.sigmoid_deriv(layer1)
            # коррекция
            self.w[1] += layer1.T.dot(layer2_delta) * self.learning_rate
            self.w[0] += layer0.T.dot(layer1_delta) * self.learning_rate
            # метрика модели
            error = np.mean(np.abs(layer2_error))
            self.errors.append(error)
            self.accuracy = (1 - error) * 100

    # сигмоида и ее производная
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return (x) * (1 - (x))
