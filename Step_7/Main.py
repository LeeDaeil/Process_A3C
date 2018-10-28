from Step_7.A3C_NETWORK import A3C_shared_network, A3C_local_network
import tensorflow as tf
from keras import backend as K
#------------------------------------------------------------------
import socket
import threading
from struct import unpack, pack
from numpy import shape
import numpy as np
from time import sleep
from collections import deque
from Step_7.Parameter import PARA
#------------------------------------------------------------------
import logging
logging.basicConfig(filename='./test.log', level=logging.DEBUG)
#------------------------------------------------------------------

episode = 0
episode_test = 0

class MainModel:
    def __init__(self):
        self.worker = []
        self._make_tensorboaed()
        global episode

        A3C = A3C_shared_network()

        for i in range(0, 2):
            self.worker.append(A3Cagent(Remote_ip=PARA.Remote_ip,
                                        Remote_port=PARA.Remote_port + i,
                                        CNS_ip=PARA.CNS_ip,
                                        CNS_port=PARA.CNS_port + i,
                                        Shared_actor_net=A3C.actor,
                                        Shared_cric_net=A3C.cric,
                                        Optimizer=A3C.optimizer,
                                        Sess=self.sess,
                                        Summary_ops=[self.summary_op, self.summary_placeholders,
                                                     self.update_ops, self.summary_writer],
                                        Test_model=False,
                                        ))
        for i in range(2, 4):
            self.worker.append(A3Cagent(Remote_ip=PARA.Remote_ip,
                                        Remote_port=PARA.Remote_port + i,
                                        CNS_ip=PARA.CNS_test_ip,
                                        CNS_port=PARA.CNS_test_port + i - 2,
                                        Shared_actor_net=A3C.actor,
                                        Shared_cric_net=A3C.cric,
                                        Optimizer=A3C.optimizer,
                                        Sess=self.sess,
                                        Summary_ops=[self.summary_op, self.summary_placeholders,
                                                     self.update_ops, self.summary_writer],
                                        Test_model=True,
                                        ))

        # 멀티프로세스 시작
        jobs =[]
        for __ in self.worker:
            __.start()
            sleep(5)

        # model save 부분
        while True:
            sleep(2)
            self._save_model(A3C)

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
        test_reward = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Test_reward', test_reward)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, test_reward]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        updata_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()

        return summary_placeholders, updata_ops, summary_op

    def _save_model(self, A3C):
        A3C.actor.save_weights("./Model/A3C_actor")
        A3C.cric.save_weights("./Model/A3C_cric")


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
                                                                           ,self.CNS_ip, self.CNS_port))

    def _init_shared_model_setting(self, Shared_actor_net, Shared_cric_net, Optimizer):
        '''
        상위 네트워크 모델의 가중치 값들을 복사해서 local network 생성
        :param Shared_actor_net: 상위 네트워크의 actor의 가중치
        :param Shared_cric_net: 상위 네트워크의 critic의 가중치
        :param Optimizer: 상위 네트워크의 옵티마이저
        :return: local-actor 와 local-critic을 생성
        '''
        A3C_local = A3C_local_network(Shared_actor_net, Shared_cric_net)
        self.local_actor_model = A3C_local.local_actor
        self.local_cric_model = A3C_local.local_cric
        self.shared_actor_net = Shared_actor_net
        self.shared_cric_net = Shared_cric_net
        self.optimizer = Optimizer

    def _init_input_window_setting(self):
        '''
        입력 윈도우의 창을 설정하는 부분
        :return: list형식의 입력 윈도우
        '''
        self.input_window_shape = self.local_cric_model.get_input_shape_at(0)
        if PARA.Model == 'LSTM':
            # (none, time-length, parameter) -> 중에서 time-length 를 반환
            self.input_window_box = deque(maxlen=self.local_cric_model.get_input_shape_at(0)[1])
        elif PARA.Model == 'DNN':
            # (none, time-length, parameter) -> 중에서 time-length 를 반환
            self.input_window_box = deque(maxlen=1)

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

    def _make_input_window(self):
        '''
        ********** 주된 편집
        :return:
        '''
        input_window_temp = [
            self.shared_mem_structure['QPROLD']['Val'],
            self.shared_mem_structure['KCNTOMS']['Val']/1000,
            # self.shared_mem_structure['KBCDO20']['Val'],
        ]
        self.input_window_box.append(input_window_temp)

        logging.debug('[{}] input_window_box_shape:{} / input_window_shape_model:{}'.format(self.name,
                                                                                        np.shape(self.input_window_box),
                                                                                        self.input_window_shape))
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

    def _init_model_information(self):
        self.avg_q_max = 0
        self.avg_loss = 0
        self.states, self.actions, self.rewards = [], [], []
        self.t_max = 3
        self.t = 0
        self.score = 0
        self.step = 0

    def _gym_send_action(self, action):
        if action == 0:
            return self._send_control_signal(['KSWO33', 'KSWO32'], [0, 0])
        elif action == 1:
            return self._send_control_signal(['KSWO33', 'KSWO32'], [1, 0])
        elif action == 2:
            return self._send_control_signal(['KSWO33', 'KSWO32'], [0, 1])

    def _gym_reward_done(self):
        if self.shared_mem_structure['QPROLD']['Val'] > 0.03:
            print('Success')
            reward = 1
        else:
            reward = 0

        if self.shared_mem_structure['KCNTOMS']['Val'] > 300:
            done = True
        else:
            done = False
        return reward, done

    def _gym_append_sample(self, input_window, policy, action, reward):
        self.states.append(input_window[0]) # (1, 2, 3) -> (2, 3) 잡아서 추출
        act = np.zeros(np.shape(policy)[0])
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    def _gym_predict_action(self, input_window):
        policy = self.local_actor_model.predict(input_window)[0]
        if self.Test_model:
            action = np.argmax(policy)
        else:
            action = np.random.choice(np.shape(policy)[0], 1, p=policy)[0]
        self.avg_q_max += np.amax(self.shared_actor_net.predict(input_window))
        return policy, action

    # ------------------------------------------------------------------
    # 네트워크 훈련 관련
    # ------------------------------------------------------------------
    def _train_model(self, done):
        # 1. Advantage 계산
        # 1.1 현재까지 수집된 보상의 크기와 동일한 배열 선언
        discount_prediction = np.zeros_like(self.rewards)
        disconnt_factor = 0.99
        running_add = 0
        if not done:
            # 만일 죽어서 업데이트가 아니라면 마지막 상태에 대한 예측 값을 계산
            running_add = self.shared_cric_net.predict(
                np.reshape(self.states[-1], (1, np.shape(self.states)[1], np.shape(self.states)[2])))[0]
        # 1.2 모든 보상에 Advantage 를 고려
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * disconnt_factor + self.rewards[t]
            discount_prediction[t] = running_add

        # 2. 모델을 훈련 시킴
        # 2.1 저장된 내용을 numpy array로 변환
        states = np.reshape(self.states, np.shape(self.states))
        # 2.2 현재 상태들을 통해서 메인 네트워크의 가치 값 계산
        values = self.shared_cric_net.predict(states)
        values = np.reshape(values, len(values))
        # 2.3 메인 네트워크 업데이트
        advantage = discount_prediction - values
        self.optimizer[0]([states, self.actions, advantage])
        self.optimizer[1]([states, discount_prediction])

        # 2.4 업데이트된 메인 네트워크를 가져와서 로컬 네트워크 업데이트
        self.local_actor_model.set_weights(self.shared_actor_net.get_weights())
        self.local_cric_model.set_weights(self.shared_cric_net.get_weights())
        print("Shared_net_work update",self)

        # 2.5 간이 저장소 초기화
        self.states, self.actions, self.rewards = [], [], []

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

        while True:
            self._update_shared_mem()
            if mode == 0:
                if self.shared_mem_structure['KCNTOMS']['Val'] < 10:
                    mode += 1
            elif mode == 1:
                if self.shared_mem_structure['KFZRUN']['Val'] == 6:
                    self._run_cns()
                if self.shared_mem_structure['KFZRUN']['Val'] == 4:
                    input_window = self._make_input_window()
                    if np.shape(input_window)[1] == 2:
                        mode += 1
                    else:
                        self._run_cns()
            elif mode == 2:
                if self.shared_mem_structure['KFZRUN']['Val'] == 4:
                    if self.shared_mem_structure['KCNTOMS']['Val'] > 30:
                        input_window = self._make_input_window()
                        mode += 1
                    else:
                        input_window = self._make_input_window()
                        self._run_cns()
            elif mode == 3:
                if self.shared_mem_structure['KFZRUN']['Val'] == 4:
                    input_window = self._make_input_window()
                    # 2.1 네트워크 액션 예측
                    policy, action = self._gym_predict_action(input_window)
                    # 2.2. 액션 전송
                    self._gym_send_action(action)
                    self._run_cns()
                    mode += 1
            elif mode == 4:
                if self.shared_mem_structure['KFZRUN']['Val'] == 4:
                    # 2.4 t+1초의 상태에 대한 보상 검증
                    reward, done = self._gym_reward_done()
                    self.score += reward
                    self.t += 1 # 모델의 업데이트 기준을 제공하기 위해서 제공
                    self.step += 1

                    # 2.5 data box 에 append
                    self._gym_append_sample(input_window, policy, action, reward)
                    logging.debug('[{}] input window\n{}'.format(self.name, input_window))
                    # 2.5.1 수집된 데이터를 일정 시간이 되면, 또는 죽으면 업데이트
                    if self.t >= self.t_max or done:
                        if self.Test_model:
                            pass
                            self.t = 0
                        else:
                            self._train_model(done)
                            self.t = 0

                    # 2.5.2 죽으면 정보 호출 및 텐서보드 업데이트
                    if done:
                        if self.Test_model:
                            episode_test += 1
                            print("Episode:{}, Score:{}, Step:{}".format(episode_test, self.score, self.step))
                            stats = [self.score]
                            self.sess.run(self.update_ops[3], feed_dict={self.summary_placeholders[3]: float(stats[0])})
                            summary_str = self.sess.run(self.summary_op)
                            self.summary_writer.add_summary(summary_str, episode_test + 1)
                            
                            if episode_test % 10 == 1:
                                # test의 네트워크를 업데이트
                                self.local_actor_model.set_weights(self.shared_actor_net.get_weights())
                                self.local_cric_model.set_weights(self.shared_cric_net.get_weights())
                                # test의 네트워크를 저장
                                self.local_actor_model.save_weights("./Test_Model/A3C_actor{}".format(episode_test))
                                self.local_cric_model.save_weights("./Test_Model/A3C_cric{}".format(episode_test))

                        else:
                            episode += 1
                            print("Episode:{}, Score:{}, Step:{}".format(episode, self.score, self.step))
                            stats = [self.score, self.avg_q_max/self.step, self.step]

                            for i in range(len(stats)):
                                self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(stats[i])})
                            summary_str = self.sess.run(self.summary_op)
                            self.summary_writer.add_summary(summary_str, episode + 1)

                        self.avg_q_max, self.avg_loss, self.score = 0, 0, 0
                        self.step = 0
                        mode += 1

                    else:
                        # 2.6 액션의 결과를 토대로 다시 업데이트
                        input_window = self._make_input_window()
                        # 2.1 네트워크 액션 예측
                        policy, action = self._gym_predict_action(input_window)
                        # 2.2. 액션 전송
                        self._gym_send_action(action)
                        self._run_cns()
            if mode == 5:
                if self.shared_mem_structure['KFZRUN']['Val'] == 6:
                    self._run_cns()
                    mode = 0
                else:
                    self._set_init_cns()
    # ------------------------------------------------------------------


if __name__ == '__main__':
    test = MainModel()