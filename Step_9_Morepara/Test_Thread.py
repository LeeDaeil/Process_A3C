import threading
import time
import datetime


class MainModel:
    def __init__(self):
        self.lock = threading.Lock()
        self.net = Network()
        self.t = datetime.datetime.now()

    def run(self):
        Worker_1(self.net, self.lock).start()
        Worker_2(self.net, self.lock).start()

        while True:
            if self.net.net_para > 100:
                e_t = datetime.datetime.now()
                print(e_t-self.t)
                time.sleep(100)


class Network:
    def __init__(self):
        self.net_para = 0


class Worker_1(threading.Thread):
    def __init__(self, network, lock):
        threading.Thread.__init__(self)
        self.network = network
        self.lock = lock

    def run(self):
        while True:
            time.sleep(1)
            # self.lock.acquire()
            temp = self.network.net_para
            time.sleep(0.1)
            self.network.net_para += 1
            print(self, temp, self.network.net_para)
            # self.lock.release()


class Worker_2(threading.Thread):
    def __init__(self, network, lock):
        threading.Thread.__init__(self)
        self.network = network
        self.lock = lock

    def run(self):
        while True:
            time.sleep(1.5)
            # self.lock.acquire()
            temp = self.network.net_para
            time.sleep(0.1)
            self.network.net_para += 10
            print(self, temp, self.network.net_para)
            # self.lock.release()


if __name__ == '__main__':
    main = MainModel()
    main.run()