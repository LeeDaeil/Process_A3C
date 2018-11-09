from Step_8.A3C_NETWORK import A3C_shared_network, A3C_local_network
import tensorflow as tf
from keras import backend as K
import gym
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras import backend as K
#------------------------------------------------------------------
import socket
import threading
from struct import unpack, pack
from numpy import shape
import numpy as np
from time import sleep
from collections import deque
from Step_8.Parameter import PARA
#------------------------------------------------------------------
import os
import shutil
#------------------------------------------------------------------
import logging
import logging.handlers
logging.basicConfig(filename='./test.log', format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
#------------------------------------------------------------------

episode = 0
episode_test = 0

class MainModel:
    def __init__(self):
        self._make_folder()
        self.worker = []
        self._make_tensorboaed()
        self.input_para = 6
        self.output_para = 3

        self.actor, self.critic = self.build_model()
        self.optimizer = [self.actor_optimizer(self.actor), self.critic_optimizer(self.critic)]



    def _run(self):
        for i in range(0, 20):
            self.worker.append(A3Cagent(Remote_ip='192.168.0.10',
                                        Remote_port=7100 + i,
                                        CNS_ip='192.168.0.2',
                                        CNS_port=7000 + i,
                                        Shared_actor_net=self.actor,
                                        Shared_cric_net=self.critic,
                                        Optimizer=self.optimizer,
                                        Sess=self.sess,
                                        Summary_ops=[self.summary_op, self.summary_placeholders,
                                                     self.update_ops, self.summary_writer],
                                        Test_model=False,
                                        ))
        # CNS2
        for i in range(0, 20):
            self.worker.append(A3Cagent(Remote_ip='192.168.0.10',
                                        Remote_port=7200 + i,
                                        CNS_ip='192.168.0.11',
                                        CNS_port=7000 + i,
                                        Shared_actor_net=self.actor,
                                        Shared_cric_net=self.critic,
                                        Optimizer=self.optimizer,
                                        Sess=self.sess,
                                        Summary_ops=[self.summary_op, self.summary_placeholders,
                                                     self.update_ops, self.summary_writer],
                                        Test_model=False,
                                        ))
        # CNS3
        for i in range(0, 20):
            self.worker.append(A3Cagent(Remote_ip='192.168.0.10',
                                        Remote_port=7300 + i,
                                        CNS_ip='192.168.0.13',
                                        CNS_port=7000 + i,
                                        Shared_actor_net=self.actor,
                                        Shared_cric_net=self.critic,
                                        Optimizer=self.optimizer,
                                        Sess=self.sess,
                                        Summary_ops=[self.summary_op, self.summary_placeholders,
                                                     self.update_ops, self.summary_writer],
                                        Test_model=False,
                                        ))

        for __ in self.worker:
           __.start()
           sleep(1)
        print("All Agent Start")

    def build_model(self):
        state = Input(batch_shape=(None, self.input_para))
        shared = Dense(40, input_dim=self.input_para, activation='relu', kernel_initializer='glorot_uniform')(state)
        # shared = Dense(48, activation='relu', kernel_initializer='glorot_uniform')(shared)

        actor_hidden = Dense(20, activation='relu', kernel_initializer='glorot_uniform')(shared)
        action_prob = Dense(self.output_para, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)

        value_hidden = Dense(9, activation='relu', kernel_initializer='he_uniform')(shared)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(value_hidden)

        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)

        actor._make_predict_function()
        critic._make_predict_function()

        with open('./Model_shape', 'w') as f:
            f.write('{}\n'.format(actor.summary()))
            f.write('{}\n'.format(critic.summary()))

        return actor, critic

    def actor_optimizer(self, actor):
        action = K.placeholder(shape=(None, self.output_para))
        advantages = K.placeholder(shape=(None, ))

        policy = actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)

        actor_loss = loss + 0.01*entropy

        optimizer = Adam(lr=0.01)
        # optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(actor.trainable_weights, [], actor_loss)
        train = K.function([actor.input, action, advantages], [], updates=updates)
        return train

    # make loss function for Value approximation
    def critic_optimizer(self, critic):
        discounted_reward = K.placeholder(shape=(None, ))

        value = critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=0.01)
        # optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(critic.trainable_weights, [], loss)
        train = K.function([critic.input, discounted_reward], [], updates=updates)
        return train

    def _save_model(self, A3C):
        A3C.actor.save_weights("./Model/A3C_actor")
        A3C.cric.save_weights("./Model/A3C_cric")

    def _make_tensorboaed(self):
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())
        self.summary_placeholders, self.update_ops, self.summary_op = self._setup_summary()
        # tensorboard dir change
        self.summary_writer = tf.summary.FileWriter('./a3c', self.sess.graph)

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
        fold_list = ['./a3c', './log', './model']
        for __ in fold_list:
            if os.path.isdir(__):
                shutil.rmtree(__)
                sleep(1)
                os.mkdir(__)
            else:
                os.mkdir(__)


class A3Cagent(threading.Thread):
    def __init__(self, Remote_ip, Remote_port, CNS_ip, CNS_port, Shared_actor_net, Shared_cric_net, Optimizer, Sess, Summary_ops, Test_model):
        threading.Thread.__init__(self)
        self.shared_mem_structure = self._make_shared_mem_structure()
        self._init_socket(Remote_ip, Remote_port, CNS_ip, CNS_port)
        self._init_shared_model_setting(Shared_actor_net, Shared_cric_net, Optimizer)
        self._init_input_window_setting()
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

        logging.debug('[{}] Initial_socket remote {}/{}, cns {}/{}'.format(self.name,
                                                                           self.Remote_ip, self.Remote_port
                                                                           , self.CNS_ip, self.CNS_port))

    def _init_shared_model_setting(self, Shared_actor_net, Shared_cric_net, Optimizer):
        self.shared_actor_net = Shared_actor_net
        self.shared_cric_net = Shared_cric_net
        self.optimizer = Optimizer

    def _init_input_window_setting(self):
        '''
        입력 윈도우의 창을 설정하는 부분
        :return: list형식의 입력 윈도우
        '''
        self.input_window_shape = self.shared_cric_net.get_input_shape_at(0)

        if PARA.Model == 'LSTM':
            # (none, time-length, parameter) -> 중에서 time-length 를 반환
            self.input_window_box = deque(maxlen=self.shared_cric_net.get_input_shape_at(0)[1])
        elif PARA.Model == 'DNN':
            # (none, time-length, parameter) -> 중에서 time-length 를 반환
            self.input_window_box = deque(maxlen=1)

    def _init_model_information(self):
        self.operation_mode = 0.4

        self.avg_q_max = 0
        self.states, self.actions, self.rewards = [], [], []
        self.action_log, self.reward_log, self.input_window_log = [], [], []
        self.score = 0
        self.step = 0

        self.update_t = 0
        self.update_t_limit = 250

        self.input_dim = 1
        self.input_number = 6

    def _init_tensorboard(self, Sess, Summary_ops):
        self.sess = Sess
        [self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer] = Summary_ops

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
    # 운전 모드 별 계산
    # ------------------------------------------------------------------

    def _calculator_operation_mode(self):
        tick = self.shared_mem_structure['KCNTOMS']['Val']
        if self.operation_mode == 0.2:      # 0.5%/min
            base_condition = tick / 60000
        elif self.operation_mode == 0.4:    # 1.0%/min
            base_condition = tick / 30000
        elif self.operation_mode == 0.6:    # 1.5%/min
            base_condition = tick / 20000
        elif self.operation_mode == 0.8:    # 2.0%/min
            base_condition = tick / 15000
        else:
            print('??')
        upper_condition = base_condition + 0.022
        stady_condition = base_condition + 0.02
        low_condition = base_condition + 0.018
        return upper_condition, stady_condition, low_condition

    def _make_input_window(self):
        power = self.shared_mem_structure['QPROREL']['Val']
        upper_condition, stady_condition, low_condition = self._calculator_operation_mode()

        input_window_temp = [
            power,
            power / 2,
            (upper_condition - power)*10,
            (power - low_condition)*10,
            stady_condition,
            self.operation_mode/10,
        ]
        if len(input_window_temp) != self.input_number:
            logging.error('[{}] _make_input_window ERROR'.format(self))
        self.input_window_box.append(input_window_temp)

        # logging.debug('[{}] input_window_box_shape:{} / input_window_shape_model:{}'.format(self.name,
        #                                                                                 np.shape(self.input_window_box),
        #                                                                                 self.input_window_shape))

        if PARA.Model == 'LSTM':
            return np.array([self.input_window_box])  # list를 np.array로 전환 (2,3) -> (1, 2, 3)
        elif PARA.Model == 'DNN':
            return np.array(self.input_window_box)  # list를 np.array로 전환 (1, 3) -> (1, 3)

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

    def _gym_send_action_append(self, parameter, value):
        for __ in range(len(parameter)):
            self.para.append(parameter[__])
            self.val.append(value[__])

    def _gym_send_action(self, action):
        power = self.shared_mem_structure['QPROREL']['Val']
        self.para = []
        self.val = []
        if self.shared_mem_structure['KLAMPO22']['Val'] == 0 and power >= 0.10:
            logging.debug('[{}] Tripblock ON\n'.format(self.name))
            self._gym_send_action_append(['KSWO22', 'KSWO21'], [1, 1])

        if power >= 0.04 and self.shared_mem_structure['KBCDO17']['Val'] <= 1800:
            logging.debug('[{}] Turbin UP {}\n'.format(self.name, self.shared_mem_structure['KBCDO17']['Val']))
            self._gym_send_action_append(['KSWO213'], [1])

        if action == 0:
            self._gym_send_action_append(['KSWO33', 'KSWO32'], [0, 0])
        elif action == 1:
            self._gym_send_action_append(['KSWO33', 'KSWO32'], [1, 0])
        elif action == 2:
            self._gym_send_action_append(['KSWO33', 'KSWO32'], [0, 1])

        return self._send_control_signal(self.para, self.val)

    def _gym_reward_done(self):
        power = self.shared_mem_structure['QPROREL']['Val']
        upper_condition, stady_condition, low_condition = self._calculator_operation_mode()

        if self.step == 600:
            reward = 0
            done = True
        else:
            if power >= low_condition and power <= upper_condition:
                reward = 1
                done = False
            else:
                reward = 0
                done = True

        return reward, done

    def _gym_append_sample(self, input_window, policy, action, reward):
        if PARA.Model == 'LSTM':
            self.states.append(input_window[0]) # (1, 2, 3) -> (2, 3) 잡아서 추출
        elif PARA.Model == 'DNN':
            self.states.append(input_window)  # (1, 2, 3) -> (2, 3) 잡아서 추출
        act = np.zeros(np.shape(policy)[0])
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    def _gym_predict_action(self, input_window):
        # policy = self.local_actor_model.predict(input_window)[0]
        policy = self.shared_actor_net.predict(np.reshape(input_window, [self.input_dim, self.input_number]))[0]
        if self.Test_model:
            # 검증 네트워크의 경우 결과를 정확하게 뱉음
            action = np.argmax(policy)
        else:
            # 훈련 네트워크의 경우 랜덤을 값을 뱉음.
            action = np.random.choice(np.shape(policy)[0], 1, p=policy)[0]
        self.avg_q_max += np.amax(self.shared_actor_net.predict(np.reshape(input_window, [self.input_dim,
                                                                                          self.input_number])))
        return policy, action

    def _gym_save_control_logger(self, input_window, action, reward):
        self.action_log.append(action)
        self.input_window_log.append(input_window)
        self.reward_log.append(reward)

    def _gym_save_control_history(self):
        if self.Test_model:
            with open('./log/Test_control_history_{}_{}.txt'.format(episode_test, self.name), 'a') as f:
                for __ in range(len(self.action_log)):
                    f.write('{}, {}, {}\n{}\n'.format(self.name, self.action_log[__], self.reward_log[__],
                                                      self.input_window_log[__]))
        else:
            with open('./log/Control_history_{}_{}.txt'.format(episode, self.name), 'a') as f:
                for __ in range(len(self.action_log)):
                    f.write('{}, {}, {}\n{}\n'.format(self.name, self.action_log[__], self.reward_log[__],
                                                      self.input_window_log[__]))

    def _gym_save_score_history(self):
        if self.Test_model:
            # 검증 네트워크에서 결과 값을 저장함
            with open('./test_history.txt', 'a') as f:
                f.writelines('{}/{}/{}\n'.format(episode_test, self.name, self.score))
        else:
            # 훈련 네트워크에서 결과 값을 저장함
            with open('./history.txt', 'a') as f:
                f.writelines('{}/{}/{}\n'.format(episode, self.name, self.score))

    # ------------------------------------------------------------------
    # 네트워크 훈련 관련
    # ------------------------------------------------------------------

    def discount_rewards(self, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.shared_cric_net.predict(np.reshape(self.states[-1], (self.input_dim,
                                                                                    self.input_number)))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * 0.99 + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # update policy network and value network every episode
    def train_episode(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)

        values = self.shared_cric_net.predict(np.array(self.states))
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        self.optimizer[0]([self.states, self.actions, advantages])
        self.optimizer[1]([self.states, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []
    # ------------------------------------------------------------------
    # 기타 편의 용
    # ------------------------------------------------------------------

    def _add_function_routine(self, input_window):
        reward = 1
        self.score += reward
        self.step += 1
        self.update_t += 1
        self._gym_save_control_logger(input_window[0], 0, reward)
        input_window = self._make_input_window()
        if PARA.show_input_windows:
            logging.debug('[{}] Input_window {}'.format(self.name, input_window))
        self._gym_send_action(0)
        self._run_cns()
        return input_window

    # ------------------------------------------------------------------
    def run(self):
        if self.Test_model:
            global episode_test
        else:
            global episode

        logging.debug('[{}] Start socket'.format(self.name))
        #
        # CNS_10_21.tar 기반의 CNS에서 구동됨.
        #
        self._set_init_cns()
        sleep(1)
        mode = 0
        while episode < 10000:
            self._update_shared_mem()
            if mode == 0:
                if self.shared_mem_structure['KCNTOMS']['Val'] < 10:
                    mode += 1
            elif mode == 1: # LSTM의 데이터를 쌓기 위해서 대기 하는 곳
                if self.shared_mem_structure['KFZRUN']['Val'] == 6:
                    self._run_cns()
                if self.shared_mem_structure['KFZRUN']['Val'] == 4:
                    input_window = self._make_input_window()
                    if np.shape(input_window)[1] == self.input_number:
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
                    input_window = self._make_input_window()
                    # 2.1 네트워크 액션 예측
                    policy, action = self._gym_predict_action(input_window[0]) #(4,)
                    # 2.2. 액션 전송
                    self._gym_send_action(action)
                    self._run_cns()
                    mode += 1
            elif mode == 4:
                if self.shared_mem_structure['KFZRUN']['Val'] == 4:
                    # 2.4 t+1초의 상태에 대한 보상 검증
                    reward, done = self._gym_reward_done()
                    self.score += reward
                    self.step += 1
                    self.update_t += 1

                    # 2.5 data box 에 append
                    self._gym_append_sample(input_window[0], policy, action, reward)
                    self._gym_save_control_logger(input_window[0], action, reward)
                    logging.debug('[{}] input window\n{}'.format(self.name, input_window[0]))

                    if self.update_t > self.update_t_limit and done != True:
                        self.train_episode(self.step != 701)
                        self.update_t = 0
                    else:
                        pass

                    # 2.5.2 죽으면 정보 호출 및 텐서보드 업데이트
                    if done:
                        # 운전 이력 저장
                        self._gym_save_score_history()

                        # 운전 목표 변화
                        # if self.operation_mode == 0.8:
                        #     self.operation_mode = 0.2   # 0.5% 출력으로 초기화
                        # elif self.operation_mode == 0.6:
                        #     self.operation_mode = 0.8
                        # elif self.operation_mode == 0.4:
                        #     self.operation_mode = 0.6
                        # elif self.operation_mode == 0.2:
                        #     self.operation_mode = 0.4

                        # if self.operation_mode == 0.4:
                        #     self.operation_mode = 0.2   # 0.5% 출력으로 초기화
                        # elif self.operation_mode == 0.2:
                        #     self.operation_mode = 0.4
                        # elif self.operation_mode == 0.4:
                        #     self.operation_mode = 0.6
                        # elif self.operation_mode == 0.2:
                        #     self.operation_mode = 0.4

                        if self.Test_model:
                            pass
                            episode_test += 1
                            self._gym_save_control_history()
                            self.action_log, self.reward_log, self.input_window_log = [], [], []
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
                            self.train_episode(self.step != 701)
                            self._gym_save_control_history()
                            self.action_log, self.reward_log, self.input_window_log = [], [], []
                            print("[TRAIN]{} Episode:{}, Score:{}, Step:{}".format(episode, self.name, self.score,
                                                                                   self.step))

                            if self.score >= 120:
                                self.score = 120
                            else:
                                pass

                            stats = [self.score, self.avg_q_max/self.step, self.step]
                            for i in range(len(stats)):
                                self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]:
                                                                                 float(stats[i])})
                            summary_str = self.sess.run(self.summary_op)
                            self.summary_writer.add_summary(summary_str, episode + 1)

                        self.avg_q_max, self.score = 0, 0
                        self.step = 0
                        self.update_t = 0

                        mode += 5
                        done = False
                    else:
                        # 2.6 액션의 결과를 토대로 다시 업데이트
                        input_window = self._make_input_window()
                        if PARA.show_input_windows:
                            logging.debug('[{}] Input_window\n{}'.format(self.name, input_window))
                        # 2.1 네트워크 액션 예측
                        policy, action = self._gym_predict_action(input_window[0])  # (4,)
                        # 2.2. 액션 전송
                        self._gym_send_action(action)
                        self._run_cns()
                        mode += 1
            if mode == 5 or mode == 6 or mode == 7:
                if self.shared_mem_structure['KFZRUN']['Val'] == 4:
                    input_window = self._add_function_routine(input_window)
                    mode += 1
            if mode == 8:
                if self.shared_mem_structure['KFZRUN']['Val'] == 4:
                    input_window = self._add_function_routine(input_window)
                    mode -= 4
            if mode == 9:
                if self.shared_mem_structure['KFZRUN']['Val'] == 6:
                    self._run_cns()
                    mode = 0
                else:
                    self._set_init_cns()
    # ------------------------------------------------------------------


if __name__ == '__main__':
    test = MainModel()
    test._run()