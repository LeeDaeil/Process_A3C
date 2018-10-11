import multiprocessing
from matplotlib import pyplot as plt
from matplotlib import animation


class Pro2(multiprocessing.Process):
    def __init__(self, shared_mem):
        multiprocessing.Process.__init__(self)

        self.shared_mem = shared_mem            # Main.py 에서 선언했던 Shared memory 를 가져옴

        self.fig = plt.figure()                 # 그래프를 그리기 위해서 matplotlib 에서 figure()을 선언함.
        self.ax = self.fig.subplots()           # figure()는 subplots()라는 하위 함수를 동봉하고 있다.
                                                # 이를 통해서 subplot 을 작성 할 수 있다.
        # matplotlib는 2단계로 구성할 수 있다.
        # 그림을 그릴때 쓰는 캔버스를 figure() 이라고 하며, 캔버스(figure())에는 하위 그래프가 그려진다.
        # 즉,
        # [ plt() ]  < - [ figure() ] <- [ ax() ]
        # 과 같은 구조를 가진다.

    def run(self):
        # matplotlib 을 통하여 애니메이션을 그리기 위해서 아래 명령어를 작성한다.
        # 이때 주의 할점은 FuncAnimation() 함수는 thread 라이브러리를 내장하고 있다.
        #
        # CPU는 하나의 코어에 2가지 쓰레드가 존재하며 ( 저가의 CPU는 쓰레드가 1개 ), 1개의 쓰레드를
        # 통해서 여러개의 프로세스를 처리 할 수 있다.
        # 1개 쓰레드는 다중 프로세스를 처리할 수 있으며, 파이썬의 경우 1개의 Process 에 다중 Thread 를
        # 선언하여 사용 할 수 있다.

        #anim = animation.FuncAnimation(self.fig, self.animate, interval=60)     # 60초 간격으로 그래프 업데이트
        anim = animation.FuncAnimation(self.fig, self.animate_ver2, interval=60)  # 60초 간격으로 그래프 업데이트
        plt.show()      # plt.show()를 통해서 animation 되고 있는 그래프 표현

    def animate(self, i):
        # animation 을 할 때는 기본적으로 2가지 과정을 거친다.

        # 1. 이전까지 그렸던 그래프를 지운다.
        self.ax.clear()

        # 2. 다시 재 빌드 한다.
        self.ax.plot(self.shared_mem['x축_데이터'], self.shared_mem['y축_데이터'])


    def animate_ver2(self, i):
        # 또한 기존의 데이터를 사용자가 원하는 입맛에 변경해서 사용도 가능하다.
        # 다음 예제코드는 사용자의 함수를 추가한 버전이다.

        # 0. 사용자가 작성한 함수를 가져오거나 작성한다.
        # 이때 주의할 점은 반드시 그래프의 x값과 y값의 길이를 동일하게 하자. 이 부분이 가장 오류가 많이
        # 유발되는 부분인 것같다.
        def user_function(mem):
            out_data = []                           # 출력 값을 생산하기 위해서 빈 리스트를 만든다.
            for i in mem['x축_데이터']:
                result = i * i/2 - 2 * i        # 사용자가 원하는 함수
                out_data.append(result)             # 계산된 결과를 출력 리스트에 append
            return out_data                         # 출력 값을 반환

        # 1. 이전까지 그렸던 그래프를 지운다.
        self.ax.clear()

        # 2. 그래프를 재 빌드 한다.
        # 이 부분은 format 합수의 특징과 ax.clear()의 특징을 생각해서 실시간으로 변화하는 값을 범례에 업데이트
        # 하는 아이디어를 구현한 것이다. 이 처럼 각 함수의 특징과 기능을 잘 생각하면 간단하고 멋진 결과를 얻을 수
        # 있다.!
        self.ax.plot(self.shared_mem['x축_데이터'], self.shared_mem['y축_데이터'], label='Test1 : {}'.format(self.shared_mem['y축_데이터'][-1]))
        # label='Test1 : {}'.format(self.shared_mem['y축_데이터'][-1])
        # 1) label = '내용'
        #   범례의 이름을 입력하는 것이다.
        # 2) label = '내용 {}'.format(변수)
        #   내용 뿐만아니라 변수 값도 출력 하는 방법이다.
        # 3) label = '내용 {}'.format(변수리스트[-1])
        #   변수 리스트의 가장 오른쪽 값, 즉 가장 마지막 데이터를 추출한다. 이 경우 그래프 가장 마지막 부분의 값을
        #   의미한다.
        out_data_list = user_function(self.shared_mem)
        self.ax.plot(self.shared_mem['x축_데이터'], out_data_list, label='Test2 : {}'.format(out_data_list[-1]))

        # 3. 범례 표시
        self.ax.legend()

        # 4. 제목 표시
        self.ax.set_title('Test graph')