import numpy as np

if __name__ == '__main__':
    list = np.array([1,2,3])
    list_data = [1,2,3]
    array = np.array(list_data)
    print(array)
    # 배열의 크기
    print(array.size)

    # 배열 내부 데이터의 타입
    print(array.dtype)

    # 인덱스로 값 뽑기
    print(array[2])

    # 배열 모양
    print(array.shape)

    # 0부터 3까지의 배열 만들기
    array1 = np.arange(4)
    print(array1)

    # 4x4의 실수형으로 0으로 찬 매트릭스 생성
    array2 = np.zeros((4, 4), dtype=float)
    print(array2)

    # 3x3의 스트링으로 1이 찬 매트릭스 생성
    array3 = np.ones((3, 3), dtype=str)
    print(array3)

    # 3x3의 배열로 0부터 9까지 랜덤하게 초기화
    array4 = np.random.randint(0, 10, (3, 3))
    print(array4)

    # 평균이 0이고, 표준편차가 1인 표준 정규를 띄는 배열(표준 정규분표)
    array5 = np.random.normal(0, 1, (3, 3))
    print(array5)

    # 배열 합치기
    array6 = np.array([1, 2, 3])
    array7 = np.array([4, 5, 6])
    array8 = np.concatenate([array6, array7])
    print(array8)

    # 배열 형태 바꾸기
    array9 = np.array([1, 2, 3, 4])
    array10 = array9.reshape((2, 2))
    print(array10)

    # 배열 세로 합치기
    array11 = np.arange(4).reshape(1, 4)
    array12 = np.arange(8).reshape(2, 4)
    array13 = np.concatenate([array11, array12], axis=0)
    print(array13)

    # 배열 나누기
    array14 = np.arange(8).reshape(2, 4)
    left, right = np.split(array14, [2], axis=1)
    # 인덱스 2를 기준으로 1 즉, 열으로 자른다.
    print(left)
    print(right)


