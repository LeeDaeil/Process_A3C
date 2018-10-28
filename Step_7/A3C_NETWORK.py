from keras.layers import Dense, Input, LSTM
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop
from keras import backend as K
from Step_6.Parameter import PARA

class A3C_net_model:
    def __init__(self):
        if PARA.Model == 'LSTM':
            self.input_shape = (2, 2)  # basic LSTM (1, 2, 3) shape
        elif PARA.Model == 'DNN':
            self.input_shape = (3,)  # basic DNN (1, 3) shape
        self.action_size = 3

    def _make_model(self):
        in_put = Input(shape=self.input_shape)

        if PARA.Model == 'LSTM':
            hidden_layer = TimeDistributed(Dense(6), input_shape=self.input_shape)(in_put)
            hidden_layer = LSTM(5, return_sequences=True)(hidden_layer)
            hidden_layer = LSTM(3)(hidden_layer)
        elif PARA.Model == 'DNN':
            hidden_layer = Dense(64, activation='relu')(in_put)
            hidden_layer = Dense(32, activation='relu')(hidden_layer)

        policy = Dense(self.action_size, activation='softmax')(hidden_layer)
        critic = Dense(1, activation='linear')(hidden_layer)

        actor = Model(inputs=in_put, outputs=policy)
        cric = Model(inputs=in_put, outputs=critic)

        return actor, cric

class A3C_shared_network:
    def __init__(self):
        print('Main_net')
        self.A3C_net_model = A3C_net_model()
        self.actor, self.cric = self._make_actor_critic_network()
        self.optimizer = [self._actor_optimizer(), self._critic_optimizer()]
        self.conter = 0

    def _make_actor_critic_network(self):
        # 네트워크를 구성하기 위해서 아래와 같이 작성한다.
        actor, cric = self.A3C_net_model._make_model()
        actor._make_predict_function()
        cric._make_predict_function()

        if PARA.show_model:
            actor.summary()
            cric.summary()

        return actor, cric

    def _actor_optimizer(self):
        action = K.placeholder(shape=[None, self.A3C_net_model.action_size])
        advantage = K.placeholder(shape=[None, ])

        policy = self.actor.output
        # 정책 크로스 엔트로피 오류함수
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantage
        cross_entropy = -K.sum(cross_entropy)

        # 탐색을 지속적으로 하기 위한 엔트로피  오류
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        # 두 오류함수를 더해 최종 오류함수를 만듬
        loss = cross_entropy + 0.01 * entropy

        optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage], [loss], updates=updates)
        return train

    def _critic_optimizer(self):
        discount_prediction = K.placeholder(shape=(None,))

        value = self.cric.output

        # [반환값 - 가치]의 제곱을 오류함수로 함.
        loss = K.mean(K.square(discount_prediction - value))

        optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.cric.trainable_weights, [], loss)
        train = K.function([self.cric.input, discount_prediction], [loss], updates=updates)
        return train

class A3C_local_network:
    def __init__(self, shared_net_actor, shared_net_cric):
        print('Local_net')
        self.A3C_net_model = A3C_net_model()
        self.local_actor, self.local_cric = self._make_local_actor_critic_network(shared_net_actor, shared_net_cric)

    def _make_local_actor_critic_network(self, shared_net_actor, shared_net_cric):
        local_actor, local_cric = self.A3C_net_model._make_model()
        local_actor._make_predict_function()
        local_cric._make_predict_function()

        local_cric.set_weights(shared_net_cric.get_weights())
        local_actor.set_weights(shared_net_actor.get_weights())

        if PARA.show_model:
            local_actor.summary()
            local_cric.summary()

        return local_actor, local_cric