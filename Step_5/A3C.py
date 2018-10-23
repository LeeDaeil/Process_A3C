#------------------------------------------------------------------
import socket
import threading
from struct import unpack, pack
from numpy import shape
import numpy as np
from time import sleep, time
#------------------------------------------------------------------
import logging
logging.basicConfig(filename='./test.log', level=logging.DEBUG)
#------------------------------------------------------------------
from Step_5.A3C_NETWORK import A3C_local_network

class A3Cagent(threading.Thread):
    def __init__(self, Remote_ip, Remote_port, CNS_ip, CNS_port, Shared_net):
        threading.Thread.__init__(self)

        logging.debug('[{}] Initial_socket'.format(self.name))

        self.Remote_ip, self.Remote_port = Remote_ip, Remote_port
        self.CNS_ip, self.CNS_port = CNS_ip, CNS_port

        self.resv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.resv_sock.bind((self.Remote_ip, self.Remote_port))

        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.shared_mem_structure = self._make_shared_mem_structure()

        self.model = Shared_net # shated_network
        self.local_model = A3C_local_network(self.model).local_model

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
        input_window = [[
            self.shared_mem_structure['KLAMPO21']['Val'],
            self.shared_mem_structure['KLAMPO22']['Val'],
            self.shared_mem_structure['KBCDO20']['Val']
        ]]
        input_window = np.array(input_window)
        logging.debug('[{}] input_window_shape:{}'.format(self.name,np.shape(input_window)))
        return input_window

    def run(self):
        logging.debug('[{}] Start socket'.format(self.name))
        #
        # CNS_10_21.tar 기반의 CNS에서 구동됨.
        #
        self._send_control_signal(['KFZRUN'], [5])  # 초기 조건을 세팅 하도록 하는 신호
        sleep(1)
        self._send_control_signal(['KFZRUN'], [3])
        while True:
            self._update_shared_mem()

            if self.shared_mem_structure['KFZRUN']['Val'] == 4: # CNS 정지가 되었다는 신호
                '''
                A3C 에이전트가 동작하는 부분이 들어 감
                1. 정지된 CNS의 현재 상태를 읽기
                2. 현재 상태에대한 A3C 에이전트가 t+1초의 액션 계산
                3. 액션을 CNS에 전송
                    self._send_control_signal(['para'], [action])
                '''
                # 1. 정지된 CNS의 현재 상태 읽기
                input_window = self._make_input_window()
                
                # 2. 네트워크 예측
                print(self.local_model.predict(input_window))

                # 4. CNS 동작
                self._send_control_signal(['KFZRUN'], [3])

            # 특정 조건 ex. 30초 넘어갈 경우 초기화
            if self.shared_mem_structure['KCNTOMS']['Val'] >= 30:
                self._send_control_signal(['KFZRUN'], [5])

            # 만약 위에서 초기화 요청을 할경우 KFZRUN의 값은 6을 반환함. 따라서 이를 감시하다가
            # KFZRUN을 3으로 바꾸면서 CNS run
            if self.shared_mem_structure['KFZRUN']['Val'] == 6:
                self._send_control_signal(['KFZRUN'], [3])

#------------------------------------------------------------------