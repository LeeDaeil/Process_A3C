import multiprocessing
from copy import deepcopy
from time import sleep


class Pro1(multiprocessing.Process):    # multiprocessing 에서 process 를 상속 받는다.
    def __init__(self, sh_mem):
        multiprocessing.Process.__init__(self)  # multiprocessing 에서 process 의 초기 조건을 가져온다.
        self.shared_mem = sh_mem                # shared memory 를 공유한다.
        self.iter = 1

    def run(self):  # 상속 받은 process 에는 run 이라는 함수를 동봉하고 있으며 자동적으로 시작한다.
        while True:
            # 1. 메모리를 업데이트 시킨다. 여기에 원하는 함수가 들어 갈 수도있고, 인공지능 방법론이 들어 갈 수 있다.
            self.update_mem()
            self.iter += 1
            # 2. 1초 대기
            sleep(3)
            print("go")

    def update_mem(self):
        # 아직 해결을 못한 문제중 하나로 메모리를 업데이트를 시도 할 때 메모리의 오염 문제를
        # 풀지 못했다. 따라서 update_mem 과 같은 함수를 만들어서 사용하고 있다.

        # 1. 기존의 공유중인 메모리의 구조를 deepcopy 를 통해서 복사한다.
        #    이를 통해서 shared memory 의 주소를 가져오지 않으면서 구조만 복사가 된다.
        temp_mem = deepcopy(self.shared_mem)

        # 2. x값, y값을 추가한다.
        temp_mem['x축_데이터'].append(self.iter)
        temp_mem['y축_데이터'].append((self.iter)*(self.iter))
        print(temp_mem['x축_데이터'][-1], temp_mem['y축_데이터'][-1], self.iter)

        # 3. 기존의 shared memory 에 overwrite 를 시킨다.
        for __ in self.shared_mem.keys():
            self.shared_mem[__] = temp_mem[__]