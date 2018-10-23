from keras.layers import Dense, Flatten, Input, LSTM
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from keras import backend as K


class A3C_net_model:
    def _make_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=3, activation='relu'))   # (1, 3)
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model


class A3C_shared_network:
    def __init__(self):
        print('Main_net')
        self.model = self._make_actor_critic_network()
    def _make_actor_critic_network(self):
        # 네트워크를 구성하기 위해서 아래와 같이 작성한다.
        model = A3C_net_model()._make_model()
        model._make_predict_function()
        return model


class A3C_local_network:
    def __init__(self, shared_net):
        print('Local_net')
        self.local_model = self._make_local_actor_critic_network(shared_net)

    def _make_local_actor_critic_network(self, shared_net):
        local_model = A3C_net_model()._make_model()
        local_model._make_predict_function()
        local_model.set_weights(shared_net.get_weights())
        return local_model