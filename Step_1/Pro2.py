import multiprocessing
from matplotlib import pyplot as plt
from matplotlib import animation

'''
수정중
'''

class Pro2(multiprocessing.Process):
    def __init__(self, shared_mem):
        multiprocessing.Process.__init__(self)

        self.shared_mem = shared_mem

        self.fig = plt.figure()
        self.ax = self.fig.subplots()

    def run(self):
        anim = animation.FuncAnimation(self.fig, self.animate, interval=60)
        plt.show()

    def animate(self, i):
        self.ax.clear()
        self.ax.plot(self.shared_mem['x축_데이터'], self.shared_mem['y축_데이터'])