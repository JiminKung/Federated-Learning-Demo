import math
import copy
import tensorflow as tf

from fed_exchange_weight_bias.Dataset import Dataset
from fed_exchange_weight_bias.Model import alexnet, scheduler


class Clients:
    def __init__(self, input_shape, classes_num, learning_rate, clients_num):
        self.current_cid = -1
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.classes_num = classes_num
        self.clients_num = clients_num
        self.dataset = Dataset(classes_num=self.classes_num,
                               split=self.clients_num,
                               one_hot=True)
        self.model = alexnet(self.input_shape, classes_num=classes_num)
        self.optimizer = tf.compat.v1.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])
        self.local_parameters_list = []
        self.initialize_local_parameters_list()

    def initialize_local_parameters_list(self):
        initial_vars = copy.deepcopy(self.model.trainable_variables)
        for _ in range(self.clients_num):
            self.local_parameters_list.append(initial_vars)

    def choose_clients(self, ratio=1.0):
        """ Randomly choose some clients """
        choose_num = math.ceil(self.clients_num * ratio)
        return list(range(choose_num))

    def download_global_gradients(self):
        pass

    def update_local_parameters(self):
        pass

    def train_local_model(self):
        pass

    def calculate_local_gradients(self):
        # save local parameters.
        pass

    def upload_local_gradients(self):
        pass
