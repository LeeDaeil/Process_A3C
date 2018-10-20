from Step_2.Loader import DataLoader
from sklearn import preprocessing


def _print_tool(orgin_data, scaled_data, scaler_name):
    # 이쁘게 결과값을 도출하기 위하여 만든 툴
    line = "="*100+"\n"
    print("{}스케일러 : {}\n{}".format(line, scaler_name, line))
    print("{}\n{}{}\n{}".format(orgin_data, line, scaled_data, line))

if __name__ == '__main__':

    # 1. Data를 읽는다.
    Data_numpy = DataLoader()._numpy_txt_read()
    # or
    # Data_numpy = DataLoader()._basic_txt_read()
    # 둘다 동일한 값을 도출한다.

    '''
    아래는 sikit-learn 을 참고하여 작성하였다.
    http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
    '''

    # 2. 단순 스케일러
    # 전체 배열을 평균값으로 정리해 버린다.
    _print_tool(Data_numpy, preprocessing.scale(Data_numpy), 'Standarzation')

    # 3.Standard scaler
    STD_scaler = preprocessing.StandardScaler().fit(Data_numpy)
    _print_tool(Data_numpy, STD_scaler.transform(Data_numpy), 'Standard scaler')

    #... 작성중