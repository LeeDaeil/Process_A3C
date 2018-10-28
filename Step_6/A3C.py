#------------------------------------------------------------------
import socket
import threading
from struct import unpack, pack
from numpy import shape
import numpy as np
from time import sleep, time
from collections import deque
from Step_6.Parameter import PARA
#------------------------------------------------------------------
import logging
logging.basicConfig(filename='./test.log', level=logging.DEBUG)
#------------------------------------------------------------------
from Step_6.A3C_NETWORK import A3C_local_network

class A3Cagent(threading.Thread):
    def __init__(self, Remote_ip, Remote_port, CNS_ip, CNS_port, Shared_actor_net, Shared_cric_net):
        threading.Thread.__init__(self)
        self.shared_mem_structure = self._make_shared_mem_structure()
        self._init_socket(Remote_ip, Remote_port, CNS_ip, CNS_port)
        self._init_shared_model_setting(Shared_actor_net, Shared_cric_net)
        self._init_input_window_setting()
        self._init_model_information()

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

        logging.debug('[{}] Initial_socket'.format(self.name))

    def _init_shared_model_setting(self, Shared_actor_net, Shared_cric_net):
        '''
        상위 네트워크 모델의 가중치 값들을 복사해서 local network 생성
        :param Shared_actor_net: 상위 네트워크의 actor의 가중치
        :param Shared_cric_net: 상위 네트워크의 critic의 가중치
        :return: local-actor 와 local-critic을 생성
        '''
        A3C_local = A3C_local_network(Shared_actor_net, Shared_cric_net)
        self.local_actor_model = A3C_local.local_actor
        self.local_cric_model = A3C_local.local_cric
        self.shared_actor_net = Shared_actor_net

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

        input_window_temp = [
            self.shared_mem_structure['KLAMPO21']['Val'],
            self.shared_mem_structure['KLAMPO22']['Val'],
            self.shared_mem_structure['KBCDO20']['Val'],
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
        self.t_max = 30
        self.t = 0
        self.score = 0

    def _gym_send_action(self, action):
        if action == 0:
            return self._send_control_signal(['KSWO33'], [0])
        elif action == 1:
            return  self._send_control_signal(['KSWO33'], [1])
        # elif action == 2:
        #     return  self._send_control_signal(['KSWO33'], [1])

    def _gym_reward_done(self):
        reward = 1
        if self.shared_mem_structure['KCNTOMS']['Val'] > 50:
            done = True
        else:
            done = False
        return reward, done

    def _gym_append_sample(self, input_window, policy, action, reward):
        self.states.append(input_window)
        act = np.zeros(np.shape(policy)[0])
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    def _gym_predict_action(self, input_window):
        policy = self.local_actor_model.predict(input_window)[0]
        action = np.random.choice(np.shape(policy)[0], 1, p=policy)[0]
        self.avg_q_max += np.amax(self.shared_actor_net.predict(input_window))
        return policy, action

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    def run(self):
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
                    # 2.5 data box 에 append
                    self._gym_append_sample(input_window, policy, action, reward)
                    if done:
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