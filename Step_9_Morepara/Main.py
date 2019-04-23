import tensorflow as tf
import pandas as pd
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

MAKE_FILE_PATH = './VER_12_LSTM'
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
        self.actor, self.critic = self.build_model(net_type='LSTM', in_pa=5, ou_pa=3, time_leg=10)
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

    def run(self):
        worker = self.build_A3C(A3C_test=False)
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
            worker.append(A3Cagent(Remote_ip='', Remote_port=7100,
                                   CNS_ip='192.168.0.2', CNS_port=7000,
                                   Shared_actor_net=self.actor, Shared_cric_net=self.critic,
                                   Optimizer=self.optimizer, Sess=self.sess,
                                   Summary_ops=[self.summary_op, self.summary_placeholders,
                                                self.update_ops, self.summary_writer],
                                   Net_type=self.net_type))
        else:
            for i in range(0, 20):
                worker.append(A3Cagent(Remote_ip='192.168.0.10', Remote_port=7100 + i,
                                       CNS_ip='192.168.0.2', CNS_port=7000 + i,
                                       Shared_actor_net=self.actor, Shared_cric_net=self.critic,
                                       Optimizer=self.optimizer, Sess=self.sess,
                                       Summary_ops=[self.summary_op, self.summary_placeholders,
                                                    self.update_ops, self.summary_writer],
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
                                       Net_type=self.net_type))
            # CNS3
            for i in range(0, 20):
                worker.append(A3Cagent(Remote_ip='192.168.0.10', Remote_port=7300 + i,
                                       CNS_ip='192.168.0.11', CNS_port=7000 + i,
                                       Shared_actor_net=self.actor, Shared_cric_net=self.critic,
                                       Optimizer=self.optimizer, Sess=self.sess,
                                       Summary_ops=[self.summary_op, self.summary_placeholders,
                                                    self.update_ops, self.summary_writer],
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
                shared = Dense(256, activation='relu')(shared)
                shared = Dense(512, activation='relu')(shared)

            elif net_type == 'CLSTM':
                shared = Conv1D(filters=10, kernel_size=3, strides=1, padding='same')(state)
                shared = MaxPooling1D(pool_size=2)(shared)
                shared = LSTM(8)(shared)

        # ----------------------------------------------------------------------------------------------------
        # Common output network
        actor_hidden = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(shared)
        action_prob = Dense(ou_pa, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)

        value_hidden = Dense(256, activation='relu', kernel_initializer='he_uniform')(shared)
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
                 Optimizer, Sess, Summary_ops, Net_type):
        threading.Thread.__init__(self)
        # CNS 환경 선언
        self.Env_CNS = CNS(CNS_ip=CNS_ip, CNS_port=CNS_port, Remote_ip=Remote_ip, Remote_port=Remote_port)
        # 내부 로거
        self.logger = logging.getLogger('{}'.format(self.name))
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler('{}/log/each_log/{}.log'.format(MAKE_FILE_PATH, self.name)))
        # 네트워크 할당 받음
        self.shared_actor_net = Shared_actor_net
        self.shared_cric_net = Shared_cric_net
        self.net_type = Net_type
        self.optimizer = Optimizer
        # 텐서보드
        self._init_tensorboard(Sess, Summary_ops)
        # 데이터 저장용
        self.DB = self.make_db()
        # 운전 모드
        self.operation_mode = 0.6
        # 운전 트리거
        self.triger = {
            'done_rod_man': 0, 'done_turbine_set': 0, 'done_steam_dump': 0, 'done_heat_drain': 0, 'done_mf_2': 0,
            'done_con_3': 0, 'done_mf_3': 0
        }


    def _init_tensorboard(self, Sess, Summary_ops):
        self.sess = Sess
        [self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer] = Summary_ops

    # ------------------------------------------------------------------
    # 운전 모드 별 계산
    # ------------------------------------------------------------------

    def calculator_operation_mode(self):
        '''
        CNS 시간과 현재 운전 목표를 고려하여 최대, 최소를 구함.
        '''
        power = self.Env_CNS.mem['QPROREL']['Val']
        tick = self.Env_CNS.mem['KCNTOMS']['Val']
        op_mode = 0.6

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
    # 현재 상태 불러 오기
    # ------------------------------------------------------------------

    def make_db(self):
        return pd.DataFrame([], columns=['Power', 'High_power', 'Low_power', 'Std_power', 'Mwe_power'])

    def make_state_data(self):

        # 입력 데이터 만들기
        up_cond, std_cond, low_cond, power = self.calculator_operation_mode()
        Mwe_power = self.Env_CNS.mem['KBCDO22']['Val'] / 1000
        self.DB.loc[len(self.DB)] = [
            power,
            (up_cond - power),
            (power - low_cond),
            std_cond,
            Mwe_power,
        ]

    def call_state(self, time_leg=1):
        self.make_state_data()
        if len(self.DB) >= time_leg:
            return self.DB.iloc[-time_leg:].values
        else:
            return 0

    def save_db(self):
        return self.DB.to_csv('./{}/{}.csv'.format(MAKE_FILE_PATH, episode))

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 네트워크 훈련 관련
    # ------------------------------------------------------------------
    
    def predict_action(self, input_window):
        predict_result = self.shared_actor_net.predict([input_window])
        policy = predict_result[0]
        action = np.random.choice(np.shape(policy)[0], 1, p=policy)[0]

        self.avg_q_max += np.amax(predict_result)
        self.average_max_step += 1      # It will be used to calculate the average_max_prob.
        return action

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
    def train_episode(self, a, r, s, done):
        discounted_rewards = self.discount_rewards(r, done)

        values = self.shared_cric_net.predict(np.array(s))
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        self.optimizer[0]([s, a, advantages])
        self.optimizer[1]([s, discounted_rewards])

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # CNS 자료 전송
    # ------------------------------------------------------------------
    
    def send_action_append(self, parameter, value):
        for __ in range(len(parameter)):
            self.para.append(parameter[__])
            self.val.append(value[__])

    def send_action(self, action):
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
            power = self.Env_CNS.mem['QPROREL']['Val'] * 100
            el_power = self.Env_CNS.mem['KBCDO22']['Val']  # elec power
            trip_b = self.Env_CNS.mem['KLAMPO22']['Val']   # Trip block condition 0 : Off, 1 : On

            turbine_set = self.Env_CNS.mem['KBCDO17']['Val']   # Turbine set point
            turbine_real = self.Env_CNS.mem['KBCDO19']['Val']   # Turbine real point
            turbine_ac = self.Env_CNS.mem['KBCDO18']['Val']     # Turbine ac condition

            load_set = self.Env_CNS.mem['KBCDO20']['Val']  # Turbine load set point
            load_rate = self.Env_CNS.mem['KBCDO21']['Val']  # Turbine load rate

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
                self.send_action_append(['KSWO22', 'KSWO21'], [1, 1])
            # Turbine set-point part
            if True:
                if power >= 4 and turbine_set < 1750:
                    self._gym_send_logger('Turbin UP {}/{}'.format(turbine_set, turbine_real))
                    self.send_action_append(['KSWO213'], [1])

                if power >= 4 and turbine_set >= 1750:
                    if self.triger['done_turbine_set'] == 0:
                        self._gym_send_logger('Turbin set point done {}/{}'.format(turbine_set, turbine_real))
                    self.triger['done_turbine_set'] = 1
                    self.send_action_append(['KSWO213'], [0])

            # Turbine acclerator up part
            if True:
                if power >= 4 and turbine_ac < 180:
                    self._gym_send_logger('Turbin ac up {}'.format(turbine_ac))
                    self.send_action_append(['KSWO215'], [1])
                elif turbine_ac >= 200:
                    self.send_action_append(['KSWO215'], [0])
            # Net break part
            if turbine_real >= 1800 and power >= 15:
                self._gym_send_logger('Net break On')
                self.send_action_append(['KSWO244'], [1])
            # Load rate control part
            if True:
                if el_power <= 0:    # before Net break - set up 100 MWe set-point
                    if power >= 10:
                        if load_set < 100:
                            self._gym_send_logger('Set point up {}'.format(load_set))
                            self.send_action_append(['KSWO225'], [1])
                        if load_set >= 100:
                            self.send_action_append(['KSWO225'], [0])
                        if load_rate < 25:
                            self._gym_send_logger('Load rate up {}'.format(load_rate))
                            self.send_action_append(['KSWO227'], [1])
                        if load_rate >= 25:
                            self.send_action_append(['KSWO227'], [0])
                else: # after Net break
                    if el_power >= 100:
                        if self.triger['done_steam_dump'] == 0:
                            self._gym_send_logger('Steam dump auto')
                        self.send_action_append(['KSWO176'], [0])
                        self.triger['done_steam_dump'] = 1
                        if self.triger['done_rod_man'] == 0:
                            self._gym_send_logger('Rod control auto')
                        self.send_action_append(['KSWO28'], [1])
                        self.triger['done_rod_man'] = 1
            # Pump part
            if True:
                if self.triger['done_rod_man'] == 1:
                    if self.triger['done_heat_drain'] == 0:
                        self._gym_send_logger('Heat drain pump on')
                    self.send_action_append(['KSWO205'], [1])
                    self.triger['done_heat_drain'] = 1

                    if el_power >= 200 and self.triger['done_heat_drain'] == 1:
                        self._gym_send_logger('Condensor Pump 2 On')
                    self.send_action_append(['KSWO205'], [1])
                    self.send_action_append(['KSWO171', 'KSWO165', 'KSWO159'], [1, 1, 1])

                    if el_power >= 400 and self.triger['done_mf_2'] == 0:
                        self._gym_send_logger('Main Feed Pump 2 On')
                    self.send_action_append(['KSWO193'], [1])
                    self.triger['done_mf_2'] = 1

                    if el_power >= 500 and self.triger['done_con_3'] == 0:
                        self._gym_send_logger('Condensor Pump 3 On')
                    self.send_action_append(['KSWO206'], [1])
                    self.triger['done_con_3'] = 1

                    if el_power >= 800 and self.triger['done_mf_3'] == 0:
                        self._gym_send_logger('Main Feed Pump 3 On')
                    self.send_action_append(['KSWO192'], [1])
                    self.triger['done_mf_3'] = 1
            # control power / Rod and Turbine load
            if True:
                if self.triger['done_rod_man'] == 1:
                    # Turbine control part Load rate control
                    if action == 0: # stay
                        self._gym_send_logger('Load stay')
                        self.send_action_append(['KSWO227', 'KSWO226'], [0, 0])
                    elif action == 1: # Up power : load rate up
                        self._gym_send_logger('Load up')
                        self.send_action_append(['KSWO227', 'KSWO226'], [1, 0])
                    elif action == 2: # Down power : load rate down
                        self._gym_send_logger('Load down')
                        self.send_action_append(['KSWO227', 'KSWO226'], [0, 1])
                else:
                    # Rod control part
                    if action == 0: # stay
                        self._gym_send_logger('Rod stay')
                        self.send_action_append(['KSWO33', 'KSWO32'], [0, 0])
                    elif action == 1: # out : increase power
                        self._gym_send_logger('Rod out')
                        self.send_action_append(['KSWO33', 'KSWO32'], [1, 0])
                    elif action == 2: # in : decrease power
                        self._gym_send_logger('Rod in')
                        self.send_action_append(['KSWO33', 'KSWO32'], [0, 1])
        self.Env_CNS.send_control_signal(self.para, self.val)

    def reward_done(self):
        up_cond, std_cond, low_cond, power = self.calculator_operation_mode()

        if self.step >= 1000:
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

    # ------------------------------------------------------------------
    # 보류
    # ------------------------------------------------------------------
    if True:
        def _gym_send_logger(self, contents):
            # contents의 내용과 공통 사항이 포함된 로거를 반환함.
            if True:
                power = self.Env_CNS.mem['QPROREL']['Val'] * 100
                el_power = self.Env_CNS.mem['KBCDO22']['Val']  # elec power
                turbine_set = self.Env_CNS.mem['KBCDO17']['Val']  # Turbine set point
                turbine_real = self.Env_CNS.mem['KBCDO19']['Val']  # Turbine real point

            step_ = '[{}]:\t'.format(self.step)
            common_ = '{}[%], {}MWe, Tru[{}/{}]'.format(power, el_power, turbine_set, turbine_real)
            return self.logger.info(step_ + contents + common_)

    def draw_img(self, now_ep):

            self.fig = plt.figure(constrained_layout=True)
            self.gs = self.fig.add_gridspec(5, 3)
            self.ax = self.fig.add_subplot(self.gs[:-2, :])
            self.ax_ = self.ax.twinx()
            # self.ax2 = self.fig.add_subplot(self.gs[-2, :])
            # self.ax3 = self.fig.add_subplot(self.gs[-1, :])

            self.ax.plot(self.DB['Power'], 'g')
            self.ax.plot(self.DB['High_power'], 'r')
            self.ax.plot(self.DB['Low_power'], 'b')

            self.ax_.plot(self.DB['Mwe_power'], 'black')

            self.ax.set_ylabel('Reactor Power [%]')
            self.ax_.set_ylabel('Electrical Power [MWe]')
            self.ax.set_ylim(bottom=0)
            self.ax_.set_ylim(bottom=0)
            self.ax.grid()
            #
            # self.ax2.plot(self.interval_log, action)
            # self.ax2.set_yticks((-1, 0, 1))
            # self.ax2.set_yticklabels(('In', 'Stay', 'Out'))
            # self.ax2.set_ylabel('Rod control')
            # self.ax2.grid()
            #
            # self.ax3.plot(self.interval_log, self.turbin_log['Setpoint'], 'r')
            # self.ax3.plot(self.interval_log, self.turbin_log['Real'], 'b')
            # self.ax3.set_yticks((900, 1800))
            # self.ax3.grid()
            # self.ax3.set_yticklabels(('900', '1800'))
            # self.ax3.set_ylabel('Turbine RPM')
            # self.ax3.set_xlabel('Time [s]')

            self.fig.savefig(fname='{}/img/{}_{}_{}.png'.format(MAKE_FILE_PATH, now_ep, self.score, self.name),
                             dpi=600, facecolor=None)

    # ------------------------------------------------------------------
    def run(self):
        # CNS_10_21.tar 기반의 CNS에서 구동됨.
        global episode, Max_score
        self.Max_score = Max_score
        logging.debug('[{}] Start socket'.format(self.name))
        self.Env_CNS.cns_init(slp=1)
        self.step, update_t = 0, 0
        self.avg_q_max, self.avg_q_max, self.score, self.average_max_step = 0, 0, 0, 0
        A_his, R_his, S_his = [], [], []
        while episode < 20000:
            now_ep = episode
            done = False
            self.start_time = datetime.datetime.now()
            # 11번 반복
            for _ in range(0, 11):
                self.Env_CNS.run_freeze()
                _ = self.call_state(time_leg=10)
                self.step += 1

            while not done:
                action = self.predict_action([self.call_state(time_leg=10)])    # -- Predic Action
                S_his.append(self.call_state(time_leg=10))                    # -- Predic Action
                A_his.append(action)                                            # -- Predic Action
                self.send_action(action)                                        # -- Send Action

                self.Env_CNS.run_freeze()                                       # -- FeedBack

                reward, done = self.reward_done()                               # -- Reward --------
                R_his.append(reward)                                            # -- Reward --------

                self.step += 1
                update_t += 1

                if update_t > 50 or done:
                    self.train_episode(a=A_his, r=R_his, s=S_his, done=done)
                    update_t = 0
                    A_his, R_his, S_his = [], [], []

                if done:
                    episode += 1
                    print("[TRAIN][{}/{}]{} Episode:{}, Score:{}, Step:{}".
                          format(self.start_time, datetime.datetime.now(), episode, self.name, self.score, self.step))
                    # 텐서보드 업데이트
                    if True:
                        stats = [self.score, self.avg_q_max / self.average_max_step, self.step]
                        for i in range(len(stats)):
                            self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]:
                                                                             float(stats[i])})
                        summary_str = self.sess.run(self.summary_op)
                        self.summary_writer.add_summary(summary_str, episode + 1)
                    # 그래프 그리기
                    if True:
                        if self.score >= 300:
                            self.draw_img(now_ep)
                    # 변수 초기화
                    if True:
                        self.step, update_t = 0, 0
                        self.avg_q_max, self.avg_q_max, self.score, self.average_max_step = 0, 0, 0, 0
                        self.triger = {
                            'done_rod_man': 0, 'done_turbine_set': 0, 'done_steam_dump': 0, 'done_heat_drain': 0,
                            'done_mf_2': 0,
                            'done_con_3': 0, 'done_mf_3': 0
                        }

            self.Env_CNS.cns_init(slp=1)
            self.save_db()
    # ------------------------------------------------------------------


class CNS:
    def __init__(self, CNS_ip, CNS_port, Remote_ip, Remote_port):
        # UDP 통신으로 저장될 메모리
        self.mem = self.make_shared_mem_structure()
        # CNS의 ip와 port번호
        self.CNS_ip, self.CNS_port = CNS_ip, CNS_port
        # UDP 수신 부분
        self.resv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.resv_sock.bind((Remote_ip, Remote_port))
        # UDP 송신 부분
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def run_freeze(self):
        self.send_control_signal(['KFZRUN'], [3])
        while True:
            self.update_shared_mem()
            if self.mem['KFZRUN']['Val'] == 6:
                self.send_control_signal(['KFZRUN'], [3])
            if self.mem['KFZRUN']['Val'] == 4:
                break
            sleep(0.5)
        return 0

    def cns_init(self, slp = 1):
        self.send_control_signal(['KFZRUN'], [5])
        while True:
            self.update_shared_mem()
            if self.mem['KFZRUN']['Val'] != 6:
                self.send_control_signal(['KFZRUN'], [5])
            if self.mem['KFZRUN']['Val'] == 6:
                break
            sleep(0.5)
        sleep(slp)
        return 0

    def send_control_signal(self, para, val):
        ''' CNS로 제어 파라메터와 값을 전송하고 0 반환'''

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

    def update_shared_mem(self):
        # UDP 통신으로 수신받은 정보를 공유 메모리에 업데이트
        data, addr = self.resv_sock.recvfrom(4008)
        for i in range(0, 4000, 20):
            sig = unpack('h', data[24+i: 26+i])[0]
            para = '12sihh' if sig == 0 else '12sfhh'
            pid, val, sig, idx = unpack(para, data[8+i:28+i])
            pid = pid.decode().rstrip('\x00') # remove '\x00'
            if pid != '':
                self.mem[pid]['Val'] = val

    def make_shared_mem_structure(self):
        # UDP 통신으로 수신된 정보를 받기위한 초기 메모리 구조 선언
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


if __name__ == '__main__':
    test = MainModel()
    test.run()