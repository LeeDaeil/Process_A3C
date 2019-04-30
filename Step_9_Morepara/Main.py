import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Input, Conv1D, MaxPooling1D, LSTM, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras import backend as K
#------------------------------------------------------------------
import socket
import threading
import datetime
import pandas as pd
from struct import unpack, pack
from numpy import shape
import numpy as np
from time import sleep
from collections import deque
from Step_9_Morepara.Parameter import PARA
#------------------------------------------------------------------
from sklearn import preprocessing
#------------------------------------------------------------------
import os
import shutil
#------------------------------------------------------------------

MAKE_FILE_PATH = './VER_3_LSTM'
os.mkdir(MAKE_FILE_PATH)

#------------------------------------------------------------------
import logging
import logging.handlers
logging.basicConfig(filename='{}/test.log'.format(MAKE_FILE_PATH), format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO)
#------------------------------------------------------------------
import matplotlib.pyplot as plt
#------------------------------------------------------------------
episode = 0
episode_test = 0
Max_score = 0       # if A3C model get Max_score, A3C model will draw the Max_score grape
FINISH_TRAIN = False
FINISH_TRAIN_CONDITION = 2.00
TEST_NETWORK = True

class MainModel:
    def __init__(self):
        global TEST_NETWORK
        self._make_folder()
        self._make_tensorboaed()
        self.actor, self.critic = self.build_model(net_type='LSTM', in_pa=4, ou_pa=3, time_leg=10)
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        self.test = TEST_NETWORK

    def run(self):
        worker = self.build_A3C(A3C_test=self.test)
        for __ in worker:
            __.start()
            sleep(1)
        print('All agent start done')
        while True:
            sleep(60)
            self._save_model()

    def build_A3C(self, A3C_test=False):
        '''
        A3C의 worker 들을 구축하는 부분
        :param A3C_test: test하는 중인지
        :return: 선언된 worker들을 반환함.
        '''
        worker = []
        if A3C_test:
            for i in range(18, 20):
                worker.append(A3Cagent(Remote_ip='192.168.0.10', Remote_port=7100 + i,
                                       CNS_ip='192.168.0.2', CNS_port=7000 + i,
                                       Shared_actor_net=self.actor, Shared_cric_net=self.critic,
                                       Optimizer=self.optimizer, Sess=self.sess,
                                       Summary_ops=[self.summary_op, self.summary_placeholders,
                                                    self.update_ops, self.summary_writer],
                                       Test_model=False,
                                       Net_type=self.net_type))
        else:
            for i in range(0, 20):
                worker.append(A3Cagent(Remote_ip='192.168.0.10', Remote_port=7100 + i,
                                       CNS_ip='192.168.0.2', CNS_port=7000 + i,
                                       Shared_actor_net=self.actor, Shared_cric_net=self.critic,
                                       Optimizer=self.optimizer, Sess=self.sess,
                                       Summary_ops=[self.summary_op, self.summary_placeholders,
                                                    self.update_ops, self.summary_writer],
                                       Test_model=False,
                                       Net_type=self.net_type))
            # CNS2
            for i in range(0, 20):
                worker.append(A3Cagent(Remote_ip='192.168.0.10', Remote_port=7200 + i,
                                       CNS_ip='192.168.0.7', CNS_port=7000 + i,
                                       Shared_actor_net=self.actor,
                                       Shared_cric_net=self.critic,
                                       Optimizer=self.optimizer,
                                       Sess=self.sess,
                                       Summary_ops=[self.summary_op, self.summary_placeholders,
                                                    self.update_ops, self.summary_writer],
                                       Test_model=False,
                                       Net_type=self.net_type))
            # CNS3
            for i in range(0, 20):
                worker.append(A3Cagent(Remote_ip='192.168.0.10', Remote_port=7300 + i,
                                       CNS_ip='192.168.0.4', CNS_port=7000 + i,
                                       Shared_actor_net=self.actor, Shared_cric_net=self.critic,
                                       Optimizer=self.optimizer, Sess=self.sess,
                                       Summary_ops=[self.summary_op, self.summary_placeholders,
                                                    self.update_ops, self.summary_writer],
                                       Test_model=False,
                                       Net_type=self.net_type))
        return worker

    def build_model(self, net_type='DNN', in_pa=1, ou_pa=1, time_leg=1):
        # 8 16 32 64 128 256 512 1024 2048
        if net_type == 'DNN':
            state = Input(batch_shape=(None, in_pa))
            shared = Dense(32, input_dim=in_pa, activation='relu', kernel_initializer='glorot_uniform')(state)
            # shared = Dense(48, activation='relu', kernel_initializer='glorot_uniform')(shared)

        elif net_type == 'CNN' or net_type == 'LSTM' or net_type == 'CLSTM':
            state = Input(batch_shape=(None, time_leg, in_pa))
            if net_type == 'CNN':
                shared = Conv1D(filters=10, kernel_size=3, strides=1, padding='same')(state)
                shared = MaxPooling1D(pool_size=2)(shared)
                shared = Flatten()(shared)

            elif net_type == 'LSTM':
                shared = LSTM(16, activation='relu')(state)
                shared = Dense(32)(shared)

            elif net_type == 'CLSTM':
                shared = Conv1D(filters=10, kernel_size=3, strides=1, padding='same')(state)
                shared = MaxPooling1D(pool_size=2)(shared)
                shared = LSTM(8)(shared)

        # ----------------------------------------------------------------------------------------------------
        # Common output network
        actor_hidden = Dense(32, activation='relu', kernel_initializer='glorot_uniform')(shared)
        action_prob = Dense(ou_pa, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)

        value_hidden = Dense(16, activation='relu', kernel_initializer='he_uniform')(shared)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(value_hidden)

        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)

        print('Make {} Network'.format(net_type))

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary(print_fn=logging.info)
        critic.summary(print_fn=logging.info)

        self.input_para = in_pa
        self.output_para = ou_pa
        self.time_leg = time_leg
        self.net_type = net_type

        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=(None, self.output_para))
        advantages = K.placeholder(shape=(None, ))

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)

        actor_loss = loss + 0.01*entropy

        # optimizer = Adam(lr=0.01)
        optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        # optimizer = Adam(lr=0.01)
        optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        return train

    def _save_model(self):
        self.actor.save_weights("{}/Model/A3C_actor.h5".format(MAKE_FILE_PATH))
        self.critic.save_weights("{}/Model/A3C_cric.h5".format(MAKE_FILE_PATH))

    def _make_tensorboaed(self):
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())
        self.summary_placeholders, self.update_ops, self.summary_op = self._setup_summary()
        # tensorboard dir change
        self.summary_writer = tf.summary.FileWriter('{}/a3c'.format(MAKE_FILE_PATH), self.sess.graph)

    def _setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        updata_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()

        return summary_placeholders, updata_ops, summary_op

    def _make_folder(self):
        fold_list = ['{}/a3c'.format(MAKE_FILE_PATH),
                     '{}/log'.format(MAKE_FILE_PATH),
                     '{}/log/each_log'.format(MAKE_FILE_PATH),
                     '{}/model'.format(MAKE_FILE_PATH),
                     '{}/img'.format(MAKE_FILE_PATH)]
        for __ in fold_list:
            if os.path.isdir(__):
                shutil.rmtree(__)
                sleep(1)
                os.mkdir(__)
            else:
                os.mkdir(__)


class A3Cagent(threading.Thread):
    def __init__(self, Remote_ip, Remote_port, CNS_ip, CNS_port, Shared_actor_net, Shared_cric_net,
                 Optimizer, Sess, Summary_ops, Test_model, Net_type):
        threading.Thread.__init__(self)
        # CNS와 통신과 데이터 교환이 가능한 모듈 호출
        self.CNS = CNS(self.name, CNS_ip, CNS_port, Remote_ip, Remote_port)
        # Network initial condition
        self.shared_actor_net = Shared_actor_net
        self.shared_cric_net = Shared_cric_net
        self.net_type = Net_type
        self.optimizer = Optimizer
        # initial input window
        self.input_window_box = deque(maxlen=self.shared_cric_net.input_shape[1])
        # Tensorboard
        self.sess = Sess
        [self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer] = Summary_ops
        # Test mode
        self.Test_model = Test_model
        # ============== 운전 모드 분당 몇% 올릴 것인지 =====
        self.operation_mode = 0.6
        # ===================================================
        # logger
        self.logger = logging.getLogger('{}'.format(self.name))
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler('{}/log/each_log/{}.log'.format(MAKE_FILE_PATH, self.name)))
        # ===================================================
        # logger 입력 윈도우 로그
        # self.logger_input = logging.getLogger('{}_input'.format(self.name))
        # self.logger_input.setLevel(logging.INFO)
        # self.logger_input.addHandler(logging.FileHandler('{}/log/{}.log'.format(MAKE_FILE_PATH, self.name)))
        # self.logger_input.info('Ep,Step,Power,High_P,Low_P,Std,Mwe')
        # ===================================================
        self.states, self.actions, self.rewards = [], [], []
        # self.action_log, self.reward_log, self.input_window_log = [], [], []
        self.total_reward, self.step, self.avg_q_max, self.average_max_step = 0, 0, 0, 0
        self.update_t, self.update_t_limit = 0, 100
        self.input_dim, self.input_number = 1, 4
        # ===================================================
        # 제어봉 로직에서 출력 로직으로 전환
        self.change_rod_to_auto = False
        # 트리거 파트
        self.triger = {
            'done_trip_block': 0, 'done_turbine_set': 0, 'done_steam_dump': 0, 'done_rod_man': 0, 'done_heat_drain': 0,
            'done_mf_2': 0, 'done_con_3': 0, 'done_mf_3': 0
        }

    # ------------------------------------------------------------------
    # 네트워크용 입력 창 생성 및 관리
    # ------------------------------------------------------------------

    def _make_input_window(self):
        if True:
            # 0. Min_max_scaler
            # min_max = preprocessing.MinMaxScaler()
            # min_data = [0.01, 0, 0, 0] #, 0]
            # max_data = [0.18, 1, 1, 1] #, 1]
            # min_max.fit([min_data, max_data])
            pass
        # 1. Read data
        up_cond, std_cond, low_cond, power = self._calculator_operation_mode()
        Mwe_power = self.CNS.mem['KBCDO22']['Val'] / 1000
        turbin_set = self.CNS.mem['KBCDO17']['Val']
        turbin_real = self.CNS.mem['KBCDO19']['Val']
        turbin_elect = self.CNS.mem['KBCDO22']['Val']
        # 2. Make (para,)
        input_window_temp = [
            power,
            (up_cond - power) * 10,
            (power - low_cond) * 10,
            std_cond,
            # Mwe_power,
        ]
        # 2.1 min_max scalling
        # input_window_temp = list(min_max.transform([input_window_temp])[0])   # 동일한 배열로 반환
        self.input_window_box.append(input_window_temp)

        if len(input_window_temp) != self.input_number:
            logging.error('[{}] _make_input_window ERROR'.format(self))
        if self.net_type == 'DNN':
            out = np.array(self.input_window_box)  # list를 np.array로 전환 (1, 3) -> (1, 3)
        else:
            out = np.array([self.input_window_box])  # list를 np.array로 전환 (2, 3) -> (1, 2, 3)

        p = ['power', 'up_cond*10', 'low_cond*10', 'std_cond', 'up_cond', 'low_cond', 'turbin_set', 'turbin_real',
             'turbin_elect', 'action']
        temp_out = out.tolist()
        for para in [up_cond, low_cond, turbin_set, turbin_real, turbin_elect]:    # 추가
            temp_out[0][-1].append(para)
        return out, temp_out

    # ------------------------------------------------------------------
    # 운전 모드 별 계산
    # ------------------------------------------------------------------

    def _calculator_operation_mode(self):
        '''
        CNS 시간과 현재 운전 목표를 고려하여 최대, 최소를 구함.
        '''
        power = self.CNS.mem['QPROREL']['Val']
        tick = self.CNS.mem['KCNTOMS']['Val']
        op_mode = self.operation_mode

        if op_mode == 0.2:      # 0.5%/min
            base_condition = tick / 60000
        elif op_mode == 0.4:    # 1.0%/min
            base_condition = tick / 30000
        elif op_mode == 0.6:    # 1.5%/min
            base_condition = tick / 20000
        elif op_mode == 0.8:    # 2.0%/min
            base_condition = tick / 15000
        else:
            print('Error calculator function')

        if self.triger['done_rod_man'] == 1:  # +- 5 % margine
            upper_condition = base_condition + 0.05
            stady_condition = base_condition + 0.02
            low_condition = base_condition - 0.01
        else: # +- 1% margine
            upper_condition = base_condition + 0.03
            stady_condition = base_condition + 0.02
            low_condition = base_condition + 0.01
        return upper_condition, stady_condition, low_condition, power

    # ------------------------------------------------------------------
    # gym
    # ------------------------------------------------------------------

    def _gym_send_logger(self, contents):
        # contents의 내용과 공통 사항이 포함된 로거를 반환함.
        if True:
            power = self.CNS.mem['QPROREL']['Val'] * 100
            el_power = self.CNS.mem['KBCDO22']['Val']  # elec power
            turbine_set = self.CNS.mem['KBCDO17']['Val']  # Turbine set point
            turbine_real = self.CNS.mem['KBCDO19']['Val']  # Turbine real point

        step_ = '[{}]:\t'.format(self.step)
        common_ = '\t{}[%], {}MWe, Tru[{}/{}]'.format(power, el_power, turbine_set, turbine_real)
        return self.logger.info(step_ + contents + common_)

    def send_act_log_append(self, parameter, value, log):
        for __ in range(len(parameter)):
            self.para.append(parameter[__])
            self.val.append(value[__])
        # if log != '':
        self._gym_send_logger(log)

    def _gym_send_action(self, action):
        '''
        :param action: 제어봉 인출 및 삽입 신호 , 터빈 load 제어 신호
        - 함수의 역할 : 입력된 액션과 현재 제어 상태를 통하여 자동화 신호들을 제작하여 CNS로 송출
        - 로직
            1) 메모리로 부터 필요한 발전소 변수 수집
            2) 수집된 변수로 자동화 로직에 따라서 제어 신호 제작 및 제어 History 작성
                * history ford : '{}/log/each_log'.format(MAKE_FILE_PATH)
            3) 제어 신호 전달
        '''
        if True:
            power = self.CNS.mem['QPROREL']['Val'] * 100
            el_power = self.CNS.mem['KBCDO22']['Val']  # elec power
            trip_b = self.CNS.mem['KLAMPO22']['Val']   # Trip block condition 0 : Off, 1 : On

            turbine_set = self.CNS.mem['KBCDO17']['Val']   # Turbine set point
            turbine_real = self.CNS.mem['KBCDO19']['Val']   # Turbine real point
            turbine_ac = self.CNS.mem['KBCDO18']['Val']     # Turbine ac condition

            load_set = self.CNS.mem['KBCDO20']['Val']  # Turbine load set point
            load_rate = self.CNS.mem['KBCDO21']['Val']  # Turbine load rate

        # 전송될 변수와 값 저장하는 리스트
        self.para = []
        self.val = []

        # History file make
        if True:
            # Trip block
            if trip_b == 0 and power >= 10:
                if self.triger['done_trip_block'] == 0:
                    self.logger.info('[{}] :\tTrip block ON'.format(self.step))
                self.triger['done_trip_block'] = 1
                self.send_act_log_append(['KSWO22', 'KSWO21'], [1, 1], log='')
            # Turbine set-point part
            if True:
                if power >= 4 and turbine_set < 1750:
                    self.send_act_log_append(['KSWO213'], [1], 'Turbin UP {}/{}'.format(turbine_set, turbine_real))

                if power >= 4 and turbine_set >= 1750:
                    if self.triger['done_turbine_set'] == 0:
                        self._gym_send_logger('Turbin set point done {}/{}'.format(turbine_set, turbine_real))
                    self.triger['done_turbine_set'] = 1
                    self.send_act_log_append(['KSWO213'], [0], log='')
            # Turbine acclerator up part
            if True:
                if power >= 4 and turbine_ac < 180:
                    self.send_act_log_append(['KSWO215'], [1], log='Turbin ac up {}'.format(turbine_ac))
                elif turbine_ac >= 200:
                    self.send_act_log_append(['KSWO215'], [0], log='')
            # Net break part
            if turbine_real >= 1800 and power >= 15:
                self.send_act_log_append(['KSWO244'], [1], 'Net break On')
            # Load rate control part
            if True:
                if el_power <= 0:    # before Net break - set up 100 MWe set-point
                    if power >= 10:
                        if load_set < 100:
                            self.send_act_log_append(['KSWO225'], [1], 'Set point up {}'.format(load_set))
                        if load_set >= 100:
                            self.send_act_log_append(['KSWO225'], [0], log='')
                        if load_rate < 25:
                            self.send_act_log_append(['KSWO227'], [1], 'Load rate up {}'.format(load_rate))
                        if load_rate >= 25:
                            self.send_act_log_append(['KSWO227'], [0], log='')
                else: # after Net break
                    if el_power >= 100:
                        if self.triger['done_steam_dump'] == 0:
                            self._gym_send_logger('Steam dump auto')
                        self.send_act_log_append(['KSWO176'], [0], log='')
                        self.triger['done_steam_dump'] = 1
                        if self.triger['done_rod_man'] == 0:
                            self._gym_send_logger('Rod control auto')
                        self.send_act_log_append(['KSWO28'], [1], log='')
                        self.triger['done_rod_man'] = 1
            # Pump part
            if True:
                if self.triger['done_rod_man'] == 1:
                    if self.triger['done_heat_drain'] == 0:
                        self._gym_send_logger('Heat drain pump on')
                    self.send_act_log_append(['KSWO205'], [1], log='')
                    self.triger['done_heat_drain'] = 1

                    if el_power >= 200 and self.triger['done_heat_drain'] == 1:
                        self._gym_send_logger('Condensor Pump 2 On')
                    self.send_act_log_append(['KSWO205'], [1],log='')
                    self.send_act_log_append(['KSWO171', 'KSWO165', 'KSWO159'], [1, 1, 1],log='')

                    if el_power >= 400 and self.triger['done_mf_2'] == 0:
                        self._gym_send_logger('Main Feed Pump 2 On')
                    self.send_act_log_append(['KSWO193'], [1],log='')
                    self.triger['done_mf_2'] = 1

                    if el_power >= 500 and self.triger['done_con_3'] == 0:
                        self._gym_send_logger('Condensor Pump 3 On')
                    self.send_act_log_append(['KSWO206'], [1],log='')
                    self.triger['done_con_3'] = 1

                    if el_power >= 800 and self.triger['done_mf_3'] == 0:
                        self._gym_send_logger('Main Feed Pump 3 On')
                    self.send_act_log_append(['KSWO192'], [1],log='')
                    self.triger['done_mf_3'] = 1
            # control power / Rod and Turbine load
            if True:
                if self.triger['done_rod_man'] == 1:
                    # Turbine control part Load rate control
                    if action == 0: # stay
                        self.send_act_log_append(['KSWO227', 'KSWO226'], [0, 0],log='Load stay')
                    elif action == 1: # Up power : load rate up
                        self.send_act_log_append(['KSWO227', 'KSWO226'], [1, 0], log='Load up')
                    elif action == 2: # Down power : load rate down
                        self.send_act_log_append(['KSWO227', 'KSWO226'], [0, 1], log='Load down')
                else:
                    # Rod control part
                    if action == 0: # stay
                        self.send_act_log_append(['KSWO33', 'KSWO32'], [0, 0], 'Rod stay')
                    elif action == 1: # out : increase power
                        self.send_act_log_append(['KSWO33', 'KSWO32'], [1, 0], 'Rod out')
                    elif action == 2: # in : decrease power
                        self.send_act_log_append(['KSWO33', 'KSWO32'], [0, 1], 'Rod in')
        self.CNS._send_control_signal(self.para, self.val)

    def _gym_reward_done(self):

        def temp_call(power, up_cond, low_cond, std_cond):
            if power > std_cond:  # std 보다 위쪽에 위치한 경우
                reward = up_cond - power  # 상위 제한치에서 파위를 빼서 보상을 지급 std에 가까울 수록 보상커짐
            elif power < std_cond:
                reward = power - low_cond
            done, score = False, 1
            return score, reward, done

        up_cond, std_cond, low_cond, power = self._calculator_operation_mode()

        if self.step >= 500:    # 제한 시간 초과 시
            score, reward, done = temp_call(power, up_cond, low_cond, std_cond)
            done = True
        else:
            if power >= low_cond and power <= up_cond: # 범위 내에서 운전 시 보상 지급
                score, reward, done = temp_call(power, up_cond, low_cond, std_cond)
            else:
                score, reward, done = 0, 0.0, True
        return score, reward, done

    def _gym_append_sample(self, input_window, policy, action, reward):
        if self.net_type == 'DNN':
            self.states.append(input_window) # (1, 2, 3) -> (2, 3) 잡아서 추출
            # print(np.shape(self.states))
        else:
            self.states.append(input_window)  # (1, 2, 3) -> (2, 3) 잡아서 추출
            # print(np.shape(self.states))
        act = np.zeros(self.shared_actor_net.output_shape[1])
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    def _gym_predict_action(self, input_window):
        # policy = self.local_actor_model.predict(input_window)[0]
        # policy = self.shared_actor_net.predict(np.reshape(input_window, [self.input_dim, self.input_number]))[0]
        predict_result = self.shared_actor_net.predict([input_window])
        policy = predict_result[0]
        if self.Test_model:
            # 검증 네트워크의 경우 결과를 정확하게 뱉음
            action = np.argmax(policy)
        elif FINISH_TRAIN:
            # if train is finish, network will be acted as "np.argmax(policy)"
            action = np.argmax(policy)
        else:
            # 훈련 네트워크의 경우 랜덤을 값을 뱉음.
            action = np.random.choice(np.shape(policy)[0], 1, p=policy)[0]
        self.avg_q_max += np.amax(predict_result)
        self.average_max_step += 1      # It will be used to calculate the average_max_prob.
        # print(predict_result, policy, action)
        return policy, action

    def draw_img(self, current_ep, data):
        # ['power', 'up_cond*10', 'low_cond*10', 'std_cond', 'up_cond', 'low_cond', 'turbin_set', 'turbin_real',
        #              'turbin_elect', 'action']
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(5, 3)
        ax = fig.add_subplot(gs[:-2, :])
        ax_ = ax.twinx()
        ax2 = fig.add_subplot(gs[-2, :])
        ax3 = fig.add_subplot(gs[-1, :])

        ax.plot(data['power'], 'g')
        ax.plot(data['low_cond'], 'r')
        ax.plot(data['up_cond'], 'b')
        ax_.plot(data['turbin_elect'], 'black')

        ax.set_ylabel('Reactor Power [%]')
        ax_.set_ylabel('Electrical Power [MWe]')
        ax.set_ylim(bottom=0)
        ax_.set_ylim(bottom=0)
        ax.grid()

        ax2.plot(data['action'])
        ax2.set_yticks((-1, 0, 1))
        ax2.set_yticklabels(('In', 'Stay', 'Out'))
        ax2.set_ylabel('Rod control')
        ax2.grid()

        ax3.plot(data['turbin_set'], 'r')
        ax3.plot(data['turbin_real'], 'b')
        ax3.set_yticks((900, 1800))
        ax3.grid()
        ax3.set_yticklabels(('900', '1800'))
        ax3.set_ylabel('Turbine RPM')
        ax3.set_xlabel('Time [s]')

        fig.savefig(fname='{}/img/{}_{}.png'.format(MAKE_FILE_PATH, current_ep, self.name), dpi=600, facecolor=None)

    # ------------------------------------------------------------------
    # 네트워크 훈련 관련
    # ------------------------------------------------------------------

    def discount_rewards(self, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            # running_add = self.shared_cric_net.predict(np.array([self.states[-1]]))[0]
            # print(self.states[-1], self)
            pass
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * 0.99 + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # update policy network and value network every episode
    def train_episode(self, done):
        global FINISH_TRAIN, FINISH_TRAIN_CONDITION
        discounted_rewards = self.discount_rewards(self.rewards, done)

        values = self.shared_cric_net.predict(np.array(self.states))
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        if FINISH_TRAIN:
            self.states, self.actions, self.rewards = [], [], []
        else:
            self.optimizer[0]([self.states, self.actions, advantages])
            self.optimizer[1]([self.states, discounted_rewards])
            self.states, self.actions, self.rewards = [], [], []

    def add_data(self, real=True, train=False):
        # current state 생성
        if True:
            if True:
                # 0. Min_max_scaler
                # min_max = preprocessing.MinMaxScaler()
                # min_data = [0.01, 0, 0, 0] #, 0]
                # max_data = [0.18, 1, 1, 1] #, 1]
                # min_max.fit([min_data, max_data])
                pass
            # 1. Read data
            up_cond, std_cond, low_cond, power = self._calculator_operation_mode()
            Mwe_power = self.CNS.mem['KBCDO22']['Val'] / 1000
            turbin_set = self.CNS.mem['KBCDO17']['Val']
            turbin_real = self.CNS.mem['KBCDO19']['Val']
            turbin_elect = self.CNS.mem['KBCDO22']['Val']

            # 2. Make (para,)
            input_window_temp = [
                power,
                (up_cond - power) * 10,
                (power - low_cond) * 10,
                std_cond,
                # Mwe_power,
            ]
        # real_db에 저장

   # ------------------------------------------------------------------
    def run(self):
        global episode

        self.CNS.init_cns()
        self.start_time = datetime.datetime.now()
        self.end_time = datetime.datetime.now()
        # 훈련 시작하는 부분
        while episode < 20000:
            # 1. 에피 소드 시작
            episode += 1            # 동시 다발적으로 발생하기 때문에 초기에 에피소드를 로드하고 +1을 함
            current_ep = episode    # 현재 에피소드의 넘버를 저장
            para = ['power', 'up_cond*10', 'low_cond*10', 'std_cond', 'up_cond', 'low_cond', 'turbin_set',
                    'turbin_real', 'turbin_elect', 'action']
            input_window_db = pd.DataFrame([], columns=para)
            self.logger.info('===Start [{}] ep=========================='.format(current_ep))
            # 2. LSTM 10 step 만큼 진행
            for i in range(0, 10):
                self.CNS.run_freeze_CNS()

                input_window, list_input_window = self._make_input_window()    # (1, 1, 4)

                list_input_window[0][-1].append(0) # 0 stay, 1: rod out, 2: rod in
                input_window_db.loc[len(input_window_db)] = list_input_window[0][-1]  # (1, 4, 4) 에서 마지막 값을 추출

            self.CNS.run_freeze_CNS()
            input_window, list_input_window = self._make_input_window()  # (1, 1, 4) # old state

            while True:
                # 오래된 상태에 대한 액션 계산
                policy, action = self._gym_predict_action(input_window)  # (1, 1, 4)
                list_input_window[0][-1].append(action)  # 0 stay, 1: rod out, 2: rod in
                input_window_db.loc[len(input_window_db)] = list_input_window[0][-1]  # (1, 4, 4) 에서 마지막 값을 추출

                # 계산된 액션을 CNS에 전송
                self._gym_send_action(action)

                # 액션을 수행하고 현재 상태 계산
                self.CNS.run_freeze_CNS()
                input_window, list_input_window = self._make_input_window()  # (1, 1, 4) # new state

                # new state에 대한 보상 계산
                score, reward, done = self._gym_reward_done()
                self._gym_append_sample(input_window[0], policy, action, reward)
                self.total_reward += reward
                self.step += 1
                self.update_t += 1

                # 게임의 종료 여부 확인
                if done or self.update_t > self.update_t_limit:
                    print('{} Train'.format(self.name))
                    self.train_episode(done)
                    self.update_t = 0
                if done:
                    self.end_time = datetime.datetime.now()
                    print("[TRAIN][{}/{}]{} Episode:{}, Score:{}, Step:{}".format(self.start_time,
                                                                                  self.end_time, self.name,
                                                                                  episode,
                                                                                  self.total_reward,
                                                                                  self.step,
                                                                                  ))
                    self.start_time = datetime.datetime.now()

                    stats = [self.total_reward, self.avg_q_max / self.average_max_step, self.step]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(stats[i])})
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode + 1)

                    # 훈련 데이터 저장
                    input_window_db.to_csv('{}/log/{}_{}.csv'.format(MAKE_FILE_PATH, self.name, current_ep))

                    # 훈련 데이터를 통한 그래프 그리기
                    self.draw_img(current_ep=current_ep, data=input_window_db)

                    # 에피소드 넘버 추가
                    episode += 1

                    self.avg_q_max, self.average_max_step, self.total_reward = 0, 0, 0
                    self.step = 0
                    self.update_t = 0
                    break
            self.CNS.init_cns()
            sleep(1)
    # ------------------------------------------------------------------


class CNS:
    def __init__(self, Thread_name, CNS_IP, CNS_Port, Remote_IP, Remote_Port):
        if True:
            # Ip, Port
            self.Remote_ip, self.Remote_port = Remote_IP, Remote_Port
            self.CNS_ip, self.CNS_port = CNS_IP, CNS_Port
            # Read Socket
            self.resv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.resv_sock.bind((self.Remote_ip, self.Remote_port))
            # Send Socket
            self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        log = '[{}] Initial_socket remote {}/{}, cns {}/{}'.format(Thread_name, self.Remote_ip, self.Remote_port,
                                                                   self.CNS_ip, self.CNS_port)
        logging.info(log)

        if True:
            # memory
            self.mem = self.make_mem_structure()

    def make_mem_structure(self):
        # 초기 shared_mem의 구조를 선언한다.
        idx = 0
        shared_mem = {}
        with open('./db.txt', 'r') as f:
            while True:
                temp_ = f.readline().split('\t')
                if temp_[0] == '':  # if empty space -> break
                    break
                sig = 0 if temp_[1] == 'INTEGER' else 1
                shared_mem[temp_[0]] = {'Sig': sig, 'Val': 0, 'Num': idx}
                idx += 1
        # 다음과정을 통하여 shared_mem 은 PID : { type. val, num }를 가진다.
        return shared_mem

    def update_mem(self):
        # binary data를 받아서 보기 쉽게 만들어서 업데이트

        data, addr = self.resv_sock.recvfrom(4008)

        for i in range(0, 4000, 20):
            sig = unpack('h', data[24 + i: 26 + i])[0]
            para = '12sihh' if sig == 0 else '12sfhh'
            pid, val, sig, idx = unpack(para, data[8 + i:28 + i])
            pid = pid.decode().rstrip('\x00')  # remove '\x00'
            if pid != '':
                self.mem[pid]['Val'] = val

    def _send_control_signal(self, para, val):
        '''
        조작 필요없음
        :param para:
        :param val:
        :return:
        '''
        for i in range(shape(para)[0]):
            self.mem[para[i]]['Val'] = val[i]
        UDP_header = b'\x00\x00\x00\x10\xa8\x0f'
        buffer = b'\x00' * 4008
        temp_data = b''

        # make temp_data to send CNS
        for i in range(shape(para)[0]):
            pid_temp = b'\x00' * 12
            pid_temp = bytes(para[i], 'ascii') + pid_temp[len(para[i]):]  # pid + \x00 ..

            para_sw = '12sihh' if self.mem[para[i]]['Sig'] == 0 else '12sfhh'

            temp_data += pack(para_sw,
                              pid_temp,
                              self.mem[para[i]]['Val'],
                              self.mem[para[i]]['Sig'],
                              self.mem[para[i]]['Num'])

        buffer = UDP_header + pack('h', shape(para)[0]) + temp_data + buffer[len(temp_data):]

        self.send_sock.sendto(buffer, (self.CNS_ip, self.CNS_port))

    def run_cns(self):
        return self._send_control_signal(['KFZRUN'], [3])

    def init_cns(self):
        # UDP 통신에 쌇인 데이터를 새롭게 하는 기능
        self._send_control_signal(['KFZRUN'], [5])
        while True:
            self.update_mem()
            if self.mem['KFZRUN']['Val'] == 6:
                # initial 상태가 완료되면 6으로 되고, break
                break
            elif self.mem['KFZRUN']['Val'] == 5:
                # 아직완료가 안된 상태
                pass
            else:
                # 4가 되는 경우: 이전의 에피소드가 끝나고 4인 상태인데
                pass
            sleep(1)

    def run_freeze_CNS(self):
        while True:
            self.update_mem()
            if self.mem['KFZRUN']['Val'] == 6:
                # initial 상태가 완료되면 6으로 되고, 1회 run 수행
                self.run_cns()
                # print('Run_cns')
            elif self.mem['KFZRUN']['Val'] == 4:
                # 1회 run 수행이 완료된 상태면 CNS run하고 break
                self.run_cns()
                break
            else:
                # initial 상태로 요청에 대한 완료가 아직 끝나지 않거나
                # 1회 run 수행이 완료가 되지 않은 상태
                pass
            sleep(1)


        # if self.mem['KFZRUN']['Val'] == 4:
        #     print('Hear')
        #     pass


if __name__ == '__main__':
    test = MainModel()
    test.run()