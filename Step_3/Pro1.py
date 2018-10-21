import multiprocessing
import time

# 이번 단계에서는 logging 모듈을 사용하여 프로그램의 로그를 저장 및 관리한다.
# 이런 로깅이 필요한 이유는 print 문보다 더욱더 선택적이고 활용성이 높다.
# 어렵다면 단순히 print 문이라고 생각하면 된다.
import logging
logging.basicConfig(filename='./test.log', level=logging.DEBUG)


# 코드의 구성은 step1과 유사하므로 설명은 생략
class Pro1(multiprocessing.Process):
    def __init__(self, shared_mem):
        multiprocessing.Process.__init__(self)

        logging.debug('[{}] start process'.format(self.name))
        self.shared_mem = shared_mem

    def run(self):
        while True:
            time.sleep(1)
            logging.debug('[{}] {}'.format(self.name, self.shared_mem.get()))