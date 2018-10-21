# step_3 내의 코드들을 불러온다. class 로 작성되어 있어서 아래와 같이 불러온다.
from Step_3.UDP_network import UDPsocket
from Step_3.Pro1 import Pro1
from Step_3.Parameter import PARA
# 멀티프로세서에서 프로세스간 자료교환을 도와주는 라이브러리 Queue 를 불러온다.
from multiprocessing import Queue


class MainModel:
    def __init__(self):
        # 이번 스텝에서는 2개의 CNS의 데이터를 받는 과정을 소개한다.
        # 이를 위해서 프로세스간 자료 공유용 메모리를 2개 작성한다.

        # 공유 메모리의 주소 명을 담기위해서 shared_mem_list 를 준비한다.
        shared_mem_list = []
        # 프로세스에 할당하기 위해서 worker 들이 들어갈 list 를 선언한다.
        self.worker = []

        for i in range(0, 2):
            # 2개 메모리 선언하고 그 주소를 shared_mem_list 에 넣는다.
            shared_mem_list.append(Queue())

            # 사전의 구성한 UDP socket 모듈을 불러와서 입력한다.
            # 기본적으로 UDPsocket(공유메모리 주소, 원격 컴퓨터의 ip, 원격 컴퓨터의 port)로 구성되어있다.
            # 포트 번호를 자동적으로 증감하기 위해서 다음과 같이 작성한다.
            # ex. 7001 -> +1 -> 7002
            self.worker.append(UDPsocket(shared_mem=shared_mem_list[i],
                                         Remote_ip=PARA.Remote_ip,
                                         Remote_port=PARA.Remote_port + i))
            # UDP 통신 소켓뿐만아니라 worker 도 추가 시켜준다.
            self.worker.append(Pro1(shared_mem=shared_mem_list[i]))

        # 멀티프로세스 시작
        # 자세한 내용은 step1 을 참조한다.
        jobs =[]
        for __ in self.worker:
            __.start()
            jobs.append(__)

        for __ in jobs:
            __.join()

if __name__ == '__main__':
    test = MainModel()