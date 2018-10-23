from Step_6.A3C import A3Cagent
from Step_6.Parameter import PARA
from Step_6.A3C_NETWORK import A3C_shared_network

class MainModel:
    def __init__(self):
        self.worker = []
        shared_model = A3C_shared_network().model

        for i in range(0, 2):
            self.worker.append(A3Cagent(Remote_ip=PARA.Remote_ip,
                                        Remote_port=PARA.Remote_port + i,
                                        CNS_ip=PARA.CNS_ip,
                                        CNS_port=PARA.CNS_port + i,
                                        Shared_net=shared_model
                                        ))

        # 멀티프로세스 시작
        jobs =[]
        for __ in self.worker:
            __.start()
            jobs.append(__)

        #for __ in jobs:
        #    __.join()


if __name__ == '__main__':
    test = MainModel()