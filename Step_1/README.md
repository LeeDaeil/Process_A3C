# Process_A3C
## Step 1 멀티프로세서를 통한 데이터공유

- 필요성:
    - 복합적인 프로그램을 작성하다보면, while 문을 개별적으로 돌리고 싶은 필요성이 발생한다.
    예를 들어, 한쪽에서는 그래프를 지속적으로 그리며, 다른 한쪽에서는 데이터 통신을 지속적으로 하는 것이다.

- 학습
    1. Multiprocessing 모듈을 이해한다.
    2. 개별적 Process 사이의 정보 교환을 위해서 Shared memory 생성 및 관리 방법을 이해한다.

- 부족사항
    1. Shared memory 관리가 어려우며 최적화가 필요하다. 관리가 어려운 이유는 프로세서가 많아지면서
       하나의 메모리를 두고 여러 프로세서가 overwrite 하기때문에 Shared memory 의 오염이 발생한다.

- Ref
    - https://docs.python.org/3.6/library/multiprocessing.html
   
### 코드 목록 및 소개
- Main.py
    - 코드의 시작

- Pro1.py
    - 1초마다 데이터를 연산하는 기능
    
- Pro2.py
    - 0.5초마다 shared memory의 값을 읽고 그래프로 그려주는 기능

