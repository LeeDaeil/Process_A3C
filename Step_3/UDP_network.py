import socket
import multiprocessing
# 이번 단계에서는 logging 모듈을 사용하여 프로그램의 로그를 저장 및 관리한다.
# 이런 로깅이 필요한 이유는 print 문보다 더욱더 선택적이고 활용성이 높다.
# 어렵다면 단순히 print 문이라고 생각하면 된다.
import logging
logging.basicConfig(filename='./test.log', level=logging.DEBUG)


class UDPsocket(multiprocessing.Process):
    def __init__(self, shared_mem, Remote_ip, Remote_port):
        multiprocessing.Process.__init__(self)

        logging.debug('[{}] Initial_socket'.format(self.name))
        # 소켓을 이란 과거에 전화에 사용되는 소켓과 동일한 의미이다.
        # 따라서 전화를 거는 방법과 동일하다.

        # 1. 전화기를 만든다.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 2. 전화기에 현재 지역번호(IP)와 나머지 번호(Port)를 입력한다.
        self.sock.bind((Remote_ip, Remote_port))

        self.shared_mem = shared_mem

    def run(self):
        logging.debug('[{}] Start socket'.format(self.name))
        while True:
            # 3. 수화기를 들고 말 또는 정보가 도착할때까지 대기한다.
            # 이때 4008은 제한치다. 예를 들어 카톡으로 파일 전송할때 몇Mb 이상 전송 불가와 동일하게
            # 받을수 있는 byte 의 수(정보의 양)가 4008 byte 라는 것이다.
            data, addr = self.sock.recvfrom(4008)

            # 출력 값이 data, addr 이다. data는 보낸 데이터를 의미하며, addr은 보낸 ip와 port의 주소이다.
            # 이를 이해하기 위해서는 상대방이 어떻게 보내는지 알아야한다.
            # UDP 통신의 경우
            # [보낸 사람의 IP][보낸 사람의 PORT][binary data.................]
            # 와 같은 구조로 가지고 있으며, 이는 네트워크 통신의 기본적인 통신 규약이다.
            print(addr)
            self.shared_mem.put(addr)

            # 다음 섹션에서는 이렇게 보내온 data(binary)를 사람이 볼수있도록 정수형으로 변환하는 과정을 보여준다.


