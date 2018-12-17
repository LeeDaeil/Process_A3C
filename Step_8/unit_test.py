import unittest
import numpy as np
from Step_8.Main import MainModel


class test_model(unittest.TestCase):
    def test_DNN_model_shape(self):
        tester = MainModel()
        actor, critic = tester.build_model(net_type='DNN', in_pa=2, ou_pa=3)
        input_array = np.array([[1, 2], [1, 1]])
        print(np.shape(input_array))
        print(actor.predict(input_array))

    def test_CNN_model_shape(self):
        tester = MainModel()
        actor, critic = tester.build_model(net_type='CNN', in_pa=2, ou_pa=3, time_leg=4)
        input_array = np.array([[[1, 1], [2, 2], [3, 3], [4, 4]]])
        print(np.shape(input_array))
        print(actor.predict(input_array))

    def test_LSTM_model_shape(self):
        tester = MainModel()
        actor, critic = tester.build_model(net_type='LSTM', in_pa=2, ou_pa=3, time_leg=4)
        input_array = np.array([[[1, 1], [2, 2], [3, 3], [4, 4]]])
        print(np.shape(input_array))
        print(actor.predict(input_array))

class test_function(unittest.TestCase):
    def test_function(self):
        self.f_1(['a', 123 +1 ])

    def f_1(self, a):
        return print(a[0], a[1])