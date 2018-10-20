import numpy


class DataLoader:
    def __init__(self, path='./test.txt'):
        self.path = path    # path 입력안할 경우 default 값으로 ./test.txt 파일 로드

    # python에서 제공하는 모듈로 작성
    def _basic_txt_read(self):
        out_ = []
        with open(self.path, 'r') as f:
            while True:
                temp = f.readline()
                if temp == '': break    # 만약 빈줄을 읽으면 마지막 줄이므로 반복 종료

                _ = []
                for __ in temp.split('\t'):
                    _.append(__.split('\n')[0]) # '\n' 엔터 키를 제외하고 분할 하기위해서 씀.

                out_.append(_)

        return numpy.array(out_[1:])

    # numpy의 툴을 활용하여 작성
    def _numpy_txt_read(self):
        return numpy.loadtxt(self.path, skiprows=1) # 위의 복잡한 로직을 한줄로 끝내버림.
        # 하지만 항상 numpy가 좋은것은 아니다. 기초 코드도 모르고 막무가내로 남발할 경우
        # 복잡한 데이터 형태를 가지는 데이터 양식은 처리할 수 없을 지도 모른다.
