# Shared memory 를 만들기 위해서 Manger 모듈 import
from multiprocessing import Manager
# Pro1 프로세스 import
from Step_1.Pro1 import Pro1
from Step_1.Pro2 import Pro2


class body:
    def __init__(self):
        # 1. Shared memory 의 구조를 선언
        self.shared_mem = self.make_memory(label=[])

    def run(self):
        process_list = [Pro1(self.shared_mem), Pro2(self.shared_mem)]       # 리스트에 실행 시키고 싶은 프로세스 넣기
        job_list = []                         # 프로세스 실행후 join 시키기 위해서 리스트 준비

        for __ in process_list:
            __.start()
            job_list.append(__)

        for job in job_list:
            job.join()

    def make_memory(self, label):
        # Tip 및 설명
        # 이 기능의 주된 역할을 Shared memory 의 구조를 작성해서 넘겨준다 (return).
        # 이렇게 하면 Shared memory 의 구조를 변경하거나 관리하기 쉽다.
        # 또한 Shared memory 의 사용 용도에 따라서 구조를 적절하게 사용하는 것이 좋으며
        # dict를 사용하는 편이 데이터 관리에 적합하다.
        return Manager().dict({                    # 기본적으로 dict의 특징을 모두 사용할 수 있다.
            'x축_데이터': [],                # 따라서 dict의 사용법을 잘 알아두는 것이 중요하다.
            'y축_데이터': [],
            1: '숫자도_입력이_가능하다.',
            'Name': 'Shared_mem1',           # 이렇게 이름으로 붙여서 메모리를 관리할 수도 있으며 응용하여 아래와 같이
        })


if __name__ == '__main__':
    Model = body()      # body class를 호출
    Model.run()         # body class의 run함수를 사용하여 프로세스 구동