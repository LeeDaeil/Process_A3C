# Process_A3C
## Step 7 A3C의 훈련 및 종료

- 필요성:
    - 

- 학습
    1. - 
    
- 부족사항
    - Thread 로 동작하지 말고 멀티프로세스에서 돌리고 싶으나 잘 안되는 것이 실정.

- 주의사항
    - CNS_10_21.tar 기반의 CNS에서 구동되며, 초기 값은 CNS 2개를 구동시켜야 코드가 원활하게 구동된다.
   
- Ref
    - 없음
   
### 코드 목록 및 소개
- Main.py
    - 코드의 시작 및 멀티프로세스의 시작
    
- A3C.py
    - A3C 에이전트가 들어있는 부분
    
- A3C_NETWORK.py
    - A3C 에이전트의 네트워크 모델이 들어 있는 부분    

### 메모
변수 증가
1. A3C_net_model , __init__ 조건에서 input shape 변경
2. _make_input_window 에서 변수 추가

### 패치노트
2018.12.17
    - LSTM, DNN, CLSTM, CNN 4개 네트워크 업데이트
    - unit test 에서 각 네트워크의 입력/출력 확인용 작성