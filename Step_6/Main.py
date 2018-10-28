from Step_6.A3C import A3Cagent
from Step_6.Parameter import PARA
from Step_6.A3C_NETWORK import A3C_shared_network
from time import  sleep

class MainModel:
    def __init__(self):
        self.worker = []
        A3C = A3C_shared_network()
        shared_actor, shared_cric = A3C.actor, A3C.cric

        for i in range(0, 2):
            self.worker.append(A3Cagent(Remote_ip=PARA.Remote_ip,
                                        Remote_port=PARA.Remote_port + i,
                                        CNS_ip=PARA.CNS_ip,
                                        CNS_port=PARA.CNS_port + i,
                                        Shared_actor_net=shared_actor,
                                        Shared_cric_net=shared_cric,
                                        ))

        # 멀티프로세스 시작
        jobs =[]
        for __ in self.worker:
            __.start()
            sleep(1)
            jobs.append(__)

if __name__ == '__main__':
    test = MainModel()