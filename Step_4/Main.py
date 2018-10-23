from Step_4.A3C import A3Cagent
from Step_4.Parameter import PARA


class MainModel:
    def __init__(self):
        self.worker = []
        for i in range(0, 2):
            self.worker.append(A3Cagent(Remote_ip=PARA.Remote_ip,
                                         Remote_port=PARA.Remote_port + i,
                                         CNS_ip=PARA.CNS_ip,
                                         CNS_port=PARA.CNS_port + i))

        # 멀티프로세스 시작
        jobs =[]
        for __ in self.worker:
            __.start()
            jobs.append(__)

        for __ in jobs:
            __.join()


if __name__ == '__main__':
    test = MainModel()