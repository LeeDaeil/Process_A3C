import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Input, Conv1D, MaxPooling1D, LSTM, Flatten
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras import backend as K
#------------------------------------------------------------------
import socket
import threading
import datetime
from struct import unpack, pack
from numpy import shape
import numpy as np
from time import sleep
from collections import deque
from Step_9_Morepara.Parameter import PARA
#------------------------------------------------------------------
import os
import shutil
#------------------------------------------------------------------

MAKE_FILE_PATH = './VER_10_LSTM'
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

class MainModel:
    def __init__(self):
        self._make_folder()
        self._make_tensorboaed()
        self.actor, self.critic = self.build_model(net_type='LSTM', in_pa=6, ou_pa=3, time_leg=10)
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        self.test = False

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
            for i in range(1, 3):
                worker.append(A3Cagent(Remote_ip='', Remote_port=7000 + i,
                                       CNS_ip='192.168.0.55', CNS_port=7000 + i,
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
                                       CNS_ip='192.168.0.11', CNS_port=7000 + i,
                                       Shared_actor_net=self.actor, Shared_cric_net=self.critic,
                                       Optimizer=self.optimizer, Sess=self.sess,
                                       Summary_ops=[self.summary_op, self.summary_placeholders,
                                                    self.update_ops, self.summary_writer],
                                       Test_model=False,
                                       Net_type=self.net_type))
        return worker

    def build_model(self, net_type='DNN', in_pa=1, ou_pa=1, time_leg=1):
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
                shared = LSTM(256, activation='relu')(state)
                shared = Dense(128, activation='relu')(shared)

            elif net_type == 'CLSTM':
                shared = Conv1D(filters=10, kernel_size=3, strides=1, padding='same')(state)
                shared = MaxPooling1D(pool_size=2)(shared)
                shared = LSTM(8)(shared)

        # ----------------------------------------------------------------------------------------------------
        # Common output network
        actor_hidden = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(shared)
        action_prob = Dense(ou_pa, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)

        value_hidden = Dense(64, activation='relu', kernel_initializer='he_uniform')(shared)
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
        self.shared_mem_structure = self._make_shared_mem_structure()
        # Network initial condition
        self.shared_actor_net = Shared_actor_net
        self.shared_cric_net = Shared_cric_net
        self.net_type = Net_type
        self.optimizer = Optimizer
        # initial socket
        self._init_socket(Remote_ip, Remote_port, CNS_ip, CNS_port)
        # initial input window
        self.input_window_box = self._make_input_window_setting(self.net_type)

        self._init_model_information()
        self._init_tensorboard(Sess, Summary_ops)
        self.Test_model = Test_model

    def _init_socket(self, Remote_ip, Remote_port, CNS_ip, CNS_port):
        '''
        :param Remote_ip: 현재 컴퓨터의 ip를 작성하는 곳
        :param Remote_port: 현재 컴퓨터의 port를 의미
        :param CNS_ip: CNS의 ip
        :param CNS_port: CNS의 port
        :return: UDP 통신을 위한 send, resv 소켓 개방 및 정보 반환
        '''
        self.Remote_ip, self.Remote_port = Remote_ip, Remote_port
        self.CNS_ip, self.CNS_port = CNS_ip, CNS_port

        self.resv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.resv_sock.bind((self.Remote_ip, self.Remote_port))

        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        logging.info('[{}] Initial_socket remote {}/{}, cns {}/{}'.format(self.name,
                                                                           self.Remote_ip, self.Remote_port
                                                                           , self.CNS_ip, self.CNS_port))

    def _init_model_information(self):
        # ============== 운전 모드 분당 몇% 올릴 것인지 =====
        self.operation_mode = 0.6
        # ===================================================
        # logger
        self.logger = logging.getLogger('{}'.format(self.name))
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler('{}/log/each_log/{}.log'.format(MAKE_FILE_PATH, self.name)))
        # ===================================================
        self.avg_q_max = 0
        self.states, self.actions, self.rewards = [], [], []
        self.action_log, self.reward_log, self.input_window_log = [], [], []
        self.score = 0
        self.step = 0
        self.average_max_step = 0

        self.update_t = 0
        self.update_t_limit = 100

        self.input_dim = 1
        self.input_number = 6

        # 제어봉 로직에서 출력 로직으로 전환
        self.change_rod_to_auto = False
        # 트리거 파트
        self.triger = {
            'done_trip_block':0, 'done_turbine_set': 0, 'done_steam_dump':0, 'done_rod_man': 0, 'done_heat_drain':0,
            'done_mf_2': 0, 'done_con_3':0, 'done_mf_3': 0
        }

        if True:
            # graphic part
            self.fig = plt.figure(constrained_layout=True)
            self.gs = self.fig.add_gridspec(5, 3)
            self.ax = self.fig.add_subplot(self.gs[:-2, :])
            self.ax_ = self.ax.twinx()
            self.ax2 = self.fig.add_subplot(self.gs[-2, :])
            self.ax3 = self.fig.add_subplot(self.gs[-1, :])

    def _init_tensorboard(self, Sess, Summary_ops):
        self.sess = Sess
        [self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer] = Summary_ops

    # ------------------------------------------------------------------
    # CNS와 통신
    # ------------------------------------------------------------------

    def _make_shared_mem_structure(self):
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

    def _update_shared_mem(self):
        '''
        조작 필요 없음
        :return:
        '''
        # binary data를 받아서 보기 쉽게 만들어서 업데이트

        data, addr = self.resv_sock.recvfrom(4008)

        for i in range(0, 4000, 20):
            sig = unpack('h', data[24+i: 26+i])[0]
            para = '12sihh' if sig == 0 else '12sfhh'
            pid, val, sig, idx = unpack(para, data[8+i:28+i])
            pid = pid.decode().rstrip('\x00') # remove '\x00'
            if pid != '':
                self.shared_mem_structure[pid]['Val'] = val

    def _send_control_signal(self, para, val):
        '''
        조작 필요없음
        :param para:
        :param val:
        :return:
        '''
        for i in range(shape(para)[0]):
            self.shared_mem_structure[para[i]]['Val'] = val[i]
        UDP_header = b'\x00\x00\x00\x10\xa8\x0f'
        buffer = b'\x00' * 4008
        temp_data = b''

        # make temp_data to send CNS
        for i in range(shape(para)[0]):
            pid_temp = b'\x00' * 12
            pid_temp = bytes(para[i], 'ascii') + pid_temp[len(para[i]):]  # pid + \x00 ..

            para_sw = '12sihh' if self.shared_mem_structure[para[i]]['Sig'] == 0 else '12sfhh'

            temp_data += pack(para_sw,
                              pid_temp,
                              self.shared_mem_structure[para[i]]['Val'],
                              self.shared_mem_structure[para[i]]['Sig'],
                              self.shared_mem_structure[para[i]]['Num'])

        buffer = UDP_header + pack('h', shape(para)[0]) + temp_data + buffer[len(temp_data):]

        self.send_sock.sendto(buffer, (self.CNS_ip, self.CNS_port))
    # ------------------------------------------------------------------
    # 네트워크용 입력 창 생성 및 관리
    # ------------------------------------------------------------------

    def _make_input_window(self):
        # 1. Read data
        up_cond, std_cond, low_cond, power = self._calculator_operation_mode()
        Mwe_power = self.shared_mem_structure['KBCDO22']['Val'] / 1000
        # 2. Make (para,)
        input_window_temp = [
            power,
            (up_cond - power) * 10,
            (power - low_cond) * 10,
            std_cond,
            Mwe_power,
            power / 2,
        ]
        # ***
        if len(input_window_temp) != self.input_number:
            logging.error('[{}] _make_input_window ERROR'.format(self))
        # 3. Append data -> (
        self.input_window_box.append(input_window_temp)
        # logging.debug('[{}] input_window_box_shape:{} / input_window_shape_model:{}'.format(self.name,
        #                                                                                 np.shape(self.input_window_box),
        #                                                                                 self.input_window_shape))
        if self.net_type == 'DNN':
            out = np.array(self.input_window_box)  # list를 np.array로 전환 (1, 3) -> (1, 3)
        else:
            out = np.array([self.input_window_box])  # list를 np.array로 전환 (2, 3) -> (1, 2, 3)
        return out

    def _make_input_window_setting(self, net_type):
        '''
        입력 윈도우의 창을 설정하는 부분
        :return: list형식의 입력 윈도우
        '''
        if net_type == 'DNN':
            # (none, time-length, parameter) -> 중에서 time-length 를 반환
            return deque(maxlen=1)
        else:
            # (none, time-length, parameter) -> 중에서 time-length 를 반환
            return deque(maxlen=self.shared_cric_net.input_shape[1])
    # ------------------------------------------------------------------
    # 운전 모드 별 계산
    # ------------------------------------------------------------------

    def _calculator_operation_mode(self):
        '''
        CNS 시간과 현재 운전 목표를 고려하여 최대, 최소를 구함.
        '''
        power = self.shared_mem_structure['QPROREL']['Val']
        tick = self.shared_mem_structure['KCNTOMS']['Val']
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
    # CNS 원격 제어 관련
    # ------------------------------------------------------------------

    def _run_cns(self):
        return self._send_control_signal(['KFZRUN'], [3])

    def _set_init_cns(self):
        return self._send_control_signal(['KFZRUN'], [5])

    # ------------------------------------------------------------------
    # gym
    # ------------------------------------------------------------------

    def _gym_send_logger(self, contents):
        # contents의 내용과 공통 사항이 포함된 로거를 반환함.
        if True:
            power = self.shared_mem_structure['QPROREL']['Val'] * 100
            el_power = self.shared_mem_structure['KBCDO22']['Val']  # elec power
            turbine_set = self.shared_mem_structure['KBCDO17']['Val']  # Turbine set point
            turbine_real = self.shared_mem_structure['KBCDO19']['Val']  # Turbine real point

        step_ = '[{}]:\t'.format(self.step)
        common_ = '{}[%], {}MWe, Tru[{}/{}]'.format(power, el_power, turbine_set, turbine_real)
        return self.logger.info(step_ + contents + common_)

    def _gym_send_action_append(self, parameter, value):
        for __ in range(len(parameter)):
            self.para.append(parameter[__])
            self.val.append(value[__])

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
            power = self.shared_mem_structure['QPROREL']['Val'] * 100
            el_power = self.shared_mem_structure['KBCDO22']['Val']  # elec power
            trip_b = self.shared_mem_structure['KLAMPO22']['Val']   # Trip block condition 0 : Off, 1 : On

            turbine_set = self.shared_mem_structure['KBCDO17']['Val']   # Turbine set point
            turbine_real = self.shared_mem_structure['KBCDO19']['Val']   # Turbine real point
            turbine_ac = self.shared_mem_structure['KBCDO18']['Val']     # Turbine ac condition

            load_set = self.shared_mem_structure['KBCDO20']['Val']  # Turbine load set point
            load_rate = self.shared_mem_structure['KBCDO21']['Val']  # Turbine load rate

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
                self._gym_send_action_append(['KSWO22', 'KSWO21'], [1, 1])
            # Turbine set-point part
            if True:
                if power >= 4 and turbine_set < 1750:
                    self._gym_send_logger('Turbin UP {}/{}'.format(turbine_set, turbine_real))
                    self._gym_send_action_append(['KSWO213'], [1])

                # if power >= 4 and turbine_set >= 1820:
                #     self.logger.info('[{}] :\tTurbin UP {}/{}'.format(self.step, turbine_set, turbine_real))
                #     self._gym_send_action_append(['KSWO212', 'KSWO213'], [1, 0])

                if power >= 4 and turbine_set >= 1750:
                    if self.triger['done_turbine_set'] == 0:
                        self._gym_send_logger('Turbin set point done {}/{}'.format(turbine_set, turbine_real))
                    self.triger['done_turbine_set'] = 1
                    self._gym_send_action_append(['KSWO213'], [0])

            # Turbine acclerator up part
            if True:
                if power >= 4 and turbine_ac < 180:
                    self._gym_send_logger('Turbin ac up {}'.format(turbine_ac))
                    self._gym_send_action_append(['KSWO215'], [1])
                elif turbine_ac >= 200:
                    self._gym_send_action_append(['KSWO215'], [0])
            # Net break part
            if turbine_real >= 1800 and power >= 15:
                self._gym_send_logger('Net break On')
                self._gym_send_action_append(['KSWO244'], [1])
            # Load rate control part
            if True:
                if el_power <= 0:    # before Net break - set up 100 MWe set-point
                    if power >= 10:
                        if load_set < 100:
                            self._gym_send_logger('Set point up {}'.format(load_set))
                            self._gym_send_action_append(['KSWO225'], [1])
                        if load_set >= 100:
                            self._gym_send_action_append(['KSWO225'], [0])
                        if load_rate < 25:
                            self._gym_send_logger('Load rate up {}'.format(load_rate))
                            self._gym_send_action_append(['KSWO227'], [1])
                        if load_rate >= 25:
                            self._gym_send_action_append(['KSWO227'], [0])
                else: # after Net break
                    if el_power >= 100:
                        if self.triger['done_steam_dump'] == 0:
                            self._gym_send_logger('Steam dump auto')
                        self._gym_send_action_append(['KSWO176'], [0])
                        self.triger['done_steam_dump'] = 1
                        if self.triger['done_rod_man'] == 0:
                            self._gym_send_logger('Rod control auto')
                        self._gym_send_action_append(['KSWO28'], [1])
                        self.triger['done_rod_man'] = 1
            # Pump part
            if True:
                if self.triger['done_rod_man'] == 1:
                    if self.triger['done_heat_drain'] == 0:
                        self._gym_send_logger('Heat drain pump on')
                    self._gym_send_action_append(['KSWO205'], [1])
                    self.triger['done_heat_drain'] = 1

                    if el_power >= 200 and self.triger['done_heat_drain'] == 1:
                        self._gym_send_logger('Condensor Pump 2 On')
                    self._gym_send_action_append(['KSWO205'], [1])
                    self._gym_send_action_append(['KSWO171', 'KSWO165', 'KSWO159'], [1, 1, 1])

                    if el_power >= 400 and self.triger['done_mf_2'] == 0:
                        self._gym_send_logger('Main Feed Pump 2 On')
                    self._gym_send_action_append(['KSWO193'], [1])
                    self.triger['done_mf_2'] = 1

                    if el_power >= 500 and self.triger['done_con_3'] == 0:
                        self._gym_send_logger('Condensor Pump 3 On')
                    self._gym_send_action_append(['KSWO206'], [1])
                    self.triger['done_con_3'] = 1

                    if el_power >= 800 and self.triger['done_mf_3'] == 0:
                        self._gym_send_logger('Main Feed Pump 3 On')
                    self._gym_send_action_append(['KSWO192'], [1])
                    self.triger['done_mf_3'] = 1
            # control power / Rod and Turbine load
            if True:
                if self.triger['done_rod_man'] == 1:
                    # Turbine control part Load rate control
                    if action == 0: # stay
                        self._gym_send_logger('Load stay')
                        self._gym_send_action_append(['KSWO227', 'KSWO226'], [0, 0])
                    elif action == 1: # Up power : load rate up
                        self._gym_send_logger('Load up')
                        self._gym_send_action_append(['KSWO227', 'KSWO226'], [1, 0])
                    elif action == 2: # Down power : load rate down
                        self._gym_send_logger('Load down')
                        self._gym_send_action_append(['KSWO227', 'KSWO226'], [0, 1])
                else:
                    # Rod control part
                    if action == 0: # stay
                        self._gym_send_logger('Rod stay')
                        self._gym_send_action_append(['KSWO33', 'KSWO32'], [0, 0])
                    elif action == 1: # out : increase power
                        self._gym_send_logger('Rod out')
                        self._gym_send_action_append(['KSWO33', 'KSWO32'], [1, 0])
                    elif action == 2: # in : decrease power
                        self._gym_send_logger('Rod in')
                        self._gym_send_action_append(['KSWO33', 'KSWO32'], [0, 1])
        self._send_control_signal(self.para, self.val)

    def _gym_reward_done(self):
        up_cond, std_cond, low_cond, power = self._calculator_operation_mode()

        if self.step >= 1200:
            reward = 0
            done = True
        else:
            if power >= low_cond and power <= up_cond:
                reward = 1
                done = False
            else:
                reward = 0
                done = True

        return reward, done

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

    def _gym_draw_img(self, max_score_ep, current_ep):
        self.ax.clear()
        self.ax_.clear()
        self.ax2.clear()
        self.ax3.clear()

        # print(self.input_window_log)
        power, low, high, action = [], [], [], []
        for __ in range(len(self.interval_log)):
            power.append((self.input_window_log[__][0]*100))
            high.append((self.input_window_log[__][0] + self.input_window_log[__][1]/10)*100)
            low.append((self.input_window_log[__][0] - self.input_window_log[__][2]/10)*100)

            if self.action_log[__] == 0:
                action.append(0)
            elif self.action_log[__] == 1:
                action.append(1)
            elif self.action_log[__] == 2:
                action.append(-1)

        # self.turbin_log = {'Setpoint': [], 'Real': [], 'Electric': []}

        self.ax.plot(self.interval_log, power, 'g')
        self.ax.plot(self.interval_log, low, 'r')
        self.ax.plot(self.interval_log, high, 'b')

        self.ax_.plot(self.interval_log, self.turbin_log['Electric'], 'black')

        self.ax.set_ylabel('Reactor Power [%]')
        self.ax_.set_ylabel('Electrical Power [MWe]')
        self.ax.set_ylim(bottom=0)
        self.ax_.set_ylim(bottom=0)
        self.ax.grid()

        self.ax2.plot(self.interval_log, action)
        self.ax2.set_yticks((-1, 0, 1))
        self.ax2.set_yticklabels(('In', 'Stay', 'Out'))
        self.ax2.set_ylabel('Rod control')
        self.ax2.grid()

        self.ax3.plot(self.interval_log, self.turbin_log['Setpoint'], 'r')
        self.ax3.plot(self.interval_log, self.turbin_log['Real'], 'b')
        self.ax3.set_yticks((900, 1800))
        self.ax3.grid()
        self.ax3.set_yticklabels(('900', '1800'))
        self.ax3.set_ylabel('Turbine RPM')
        self.ax3.set_xlabel('Time [s]')

        self.fig.savefig(fname='{}/img/{}_{}_{}.png'.format(MAKE_FILE_PATH, current_ep-1, max_score_ep, self.name),
                         dpi=600, facecolor=None)

    def _gym_save_control_logger(self, input_window, action, reward):
        self.interval_log.append(self.interval)
        self.interval += 1
        self.action_log.append(action)
        if self.net_type == 'DNN':
            self.input_window_log.append(input_window)
        else:
            self.input_window_log.append(input_window[-1])

        self.turbin_log['Setpoint'].append(self.shared_mem_structure['KBCDO17']['Val'])
        self.turbin_log['Real'].append(self.shared_mem_structure['KBCDO19']['Val'])
        self.turbin_log['Electric'].append(self.shared_mem_structure['KBCDO22']['Val'])

        self.reward_log.append(reward)

    def _gym_save_control_history(self):
        if self.Test_model:
            with open('{}/log/Test_control_history_{}_{}.txt'.format(MAKE_FILE_PATH,
                                                                     episode_test, self.name), 'a') as f:
                for __ in range(len(self.action_log)):
                    f.write('{}, {}, {}, {}, '.format(self.name,
                                                      self.interval_log[__],
                                                      self.reward_log[__],
                                                      self.action_log[__]))
                    for _ in self.input_window_log[__]:
                        f.write('{}, '.format(self.input_window_log[__][_]))
                    f.write('\n')
        else:
            with open('{}/log/Control_history_{}_{}.txt'.format(MAKE_FILE_PATH, episode, self.name), 'a') as f:
                for __ in range(len(self.action_log)):
                    f.write('{}, {}, {}, {}, '.format(self.name,
                                                      self.interval_log[__],
                                                      self.reward_log[__],
                                                      self.action_log[__]))
                    for _ in self.input_window_log[__]:
                        f.write('{}, '.format(_))
                    f.write('\n')

    def _gym_save_score_history(self):
        if self.Test_model:
            # 검증 네트워크에서 결과 값을 저장함
            with open('{}/test_history.txt'.format(MAKE_FILE_PATH), 'a') as f:
                f.writelines('{}/{}/{}\n'.format(episode_test, self.name, self.score))
        else:
            # 훈련 네트워크에서 결과 값을 저장함
            with open('{}/history.txt'.format(MAKE_FILE_PATH), 'a') as f:
                f.writelines('{}/{}/{}\n'.format(episode, self.name, self.score))

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
    # ------------------------------------------------------------------
    # 기타 편의 용
    # ------------------------------------------------------------------

    def _add_function_routine(self, input_window, action):
        reward = 1
        self.score += reward
        self.step += 1
        self.update_t += 1
        self._gym_save_control_logger(input_window[0], action, reward)
        input_window = self._make_input_window()
        if PARA.show_input_windows:
            logging.debug('[{}] Input_window {}'.format(self.name, input_window))
        self._gym_send_action(action)
        self._run_cns()
        return input_window

    # ------------------------------------------------------------------
    def run(self):
        if self.Test_model:
            global episode_test
        else:
            global episode

        global Max_score, FINISH_TRAIN, FINISH_TRAIN_CONDITION
        self.Max_score = Max_score

        logging.debug('[{}] Start socket'.format(self.name))
        self.action_log, self.reward_log, self.input_window_log, self.interval_log, self.interval = [], [], [], [], 0
        self.turbin_log = {'Setpoint': [], 'Real': [], 'Electric': []}
        #
        # CNS_10_21.tar 기반의 CNS에서 구동됨.
        #
        self._set_init_cns()
        sleep(1)
        mode = 0

        self.start_time = datetime.datetime.now()
        self.end_time = datetime.datetime.now()
        while episode < 20000:

            self._update_shared_mem()
            if mode == 0:
                if self.shared_mem_structure['KCNTOMS']['Val'] < 10:
                    mode += 1
            elif mode == 1: # LSTM의 데이터를 쌓기 위해서 대기 하는 곳
                # print('Mode1')
                if self.shared_mem_structure['KFZRUN']['Val'] == 6:
                    self._run_cns()
                if self.shared_mem_structure['KFZRUN']['Val'] == 4:
                    input_window = self._make_input_window()
                    if self.net_type == 'DNN':
                        mode += 1 # DNN인 경우 pass
                    else:
                        if np.shape(input_window)[1] == self.shared_cric_net.input_shape[1]:
                            # CNN, LSTM, C_LSTM에 입력 shape와 동일 할 때까지 대기
                            mode += 1
                        else:
                            self._run_cns()
            elif mode == 2: # 좀 더 대기하는 곳
                if self.shared_mem_structure['KFZRUN']['Val'] == 4:
                    if self.shared_mem_structure['KCNTOMS']['Val'] > 15:
                        input_window = self._make_input_window()
                        mode += 1
                    else:
                        input_window = self._make_input_window()
                        self._run_cns()
            elif mode == 3: # 초기 액션
                if self.shared_mem_structure['KFZRUN']['Val'] == 4:

                    self.logger.info('===Start [{}] ep=========================='.format(episode))

                    input_window = self._make_input_window()
                    # 2.1 네트워크 액션 예측
                    policy, action = self._gym_predict_action(input_window) #(4,)
                    # 2.2. 액션 전송
                    self._gym_send_action(action)
                    self._run_cns()
                    mode += 1
            elif mode == 4:
                if self.shared_mem_structure['KFZRUN']['Val'] == 4:
                    # 2.4 t+1초의 상태에 대한 보상 검증
                    reward, done = self._gym_reward_done()
                    if reward == 100:
                        self.score += 1
                    else:
                        self.score += reward

                    self.step += 1
                    self.update_t += 1

                    # 2.5 data box 에 append
                    self._gym_append_sample(input_window[0], policy, action, reward)
                    self._gym_save_control_logger(input_window[0], action, reward)
                    if PARA.save_input_log:
                        logging.debug('[{}] input window\n{}'.format(self.name, input_window[0]))

                    if self.update_t > self.update_t_limit or done:
                        print('{} Train'.format(self.name))
                        self.train_episode(done)
                        self.update_t = 0
                    else:
                        pass

                    # 2.5.2 죽으면 정보 호출 및 텐서보드 업데이트
                    if done:
                        # 운전 이력 저장
                        self._gym_save_score_history()

                        if self.Test_model:
                            pass
                            episode_test += 1
                            self._gym_save_control_history()
                            self.action_log, self.reward_log, self.input_window_log, self.interval_log, self.interval = [], [], [], [], 0
                            self.turbin_log = {'Setpoint': [], 'Real': [], 'Electric': []}
                            print("[TEST]{} Test_Episode:{}, Score:{}, Step:{}".format(episode_test, self.name, self.score,
                                                                                  self.step))
                            # if episode_test % 10 == 1:
                            #     logging.debug('[{}] [TEST] Shared_net_work update'.format(self.name))
                            #     # test의 네트워크를 업데이트
                            #     # self.local_actor_model.set_weights(self.shared_actor_net.get_weights())
                            #     # self.local_cric_model.set_weights(self.shared_cric_net.get_weights())
                            #     self.local_actor_model.load_weights('./Model/A3C_actor')
                            #     self.local_cric_model.load_weights('./Model/A3C_cric')
                            #     # test의 네트워크를 저장
                            #     self.local_actor_model.save_weights("./Test_Model/{}_A3C_actor".format(episode))
                            #     self.local_cric_model.save_weights("./Test_Model/{}_A3C_cric".format(episode))
                            #     self.states, self.actions, self.rewards = [], [], []

                        else:
                            episode += 1
                            # self.train_episode(self.step != 1201)
                            self._gym_save_control_history()

                            if self.score >= Max_score or self.score >= 300:
                                self._gym_draw_img(current_ep=episode, max_score_ep=self.score)
                                Max_score = self.score
                                self.Max_score = Max_score

                            self.action_log, self.reward_log, self.input_window_log, self.interval_log, self.interval = [], [], [], [], 0

                            self.end_time = datetime.datetime.now()
                            self.turbin_log = {'Setpoint': [], 'Real': [], 'Electric': []}
                            print("[TRAIN][{}/{}]{} Episode:{}, Score:{}, Step:{}".format(self.start_time,
                                                                                          self.end_time,
                                                                                          episode, self.name,
                                                                                          self.score, self.step))
                            self.start_time = datetime.datetime.now()

                            # if self.score >= 600:
                            #     self.score = 600
                            # else:
                            #     pass

                            stats = [self.score, self.avg_q_max/self.average_max_step, self.step]
                            for i in range(len(stats)):
                                self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]:
                                                                                 float(stats[i])})
                            summary_str = self.sess.run(self.summary_op)
                            self.summary_writer.add_summary(summary_str, episode + 1)

                        if FINISH_TRAIN != True:
                            if (self.avg_q_max/self.average_max_step) >= FINISH_TRAIN_CONDITION:
                                print("[FINISH]{} Episode:{}".format(episode, self.name))
                                FINISH_TRAIN = True

                        self.avg_q_max, self.average_max_step, self.score = 0, 0, 0
                        self.step = 0
                        self.update_t = 0

                        mode += 5
                        done = False
                    else:
                        # 2.6 액션의 결과를 토대로 다시 업데이트
                        input_window = self._make_input_window()
                        if PARA.show_input_windows:
                            logging.debug('[{}] Input_window {}'.format(self.name, input_window))
                        # 2.1 네트워크 액션 예측
                        policy, action = self._gym_predict_action(input_window)
                        # 2.2. 액션 전송
                        self._gym_send_action(action)
                        self._run_cns()
                        mode += 1

            if mode == 5 or mode == 6 or mode == 7:
                if self.shared_mem_structure['KFZRUN']['Val'] == 4:
                    input_window = self._add_function_routine(input_window, action)
                    mode += 1
            if mode == 8:
                if self.shared_mem_structure['KFZRUN']['Val'] == 4:
                    input_window = self._add_function_routine(input_window, action)
                    mode -= 4
            if mode == 9:
                if self.shared_mem_structure['KFZRUN']['Val'] == 6:
                    self._run_cns()
                    mode = 0
                else:
                    # initial input window
                    self.input_window_box = self._make_input_window_setting(self.net_type)
                    self._set_init_cns()
    # ------------------------------------------------------------------


if __name__ == '__main__':
    test = MainModel()
    test.run()