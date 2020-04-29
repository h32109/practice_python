from __future__ import print_function
import torch

if __name__ == '__main__':
    # 초기화 되지 않은 5x3행렬 생성
    x = torch.empty(5, 3)
    print(x)

    # 무작위 초기화 행렬
    x = torch.rand(5, 3)
    print(x)

    # long type의 0으로 초기화된 5x3 행렬
    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)

    # 데이터로부터 tensor 직접 생성
    x = torch.tensor([5.5, 3])
    print(x)

    # tesnsor 바탕으로 tensor를 생성
    x = x.new_ones(5, 3, dtype=torch.double)

    # dtype을 오버라이드하고 동일한 크기를 가진다.
    x = torch.randn_like(x, dtype=torch.float)

    # 행렬의 크기 size는 튜플과 같으며 모든 튜플 연산을 제공한다.
    print(x.size())

    # 연산 지원
    y = torch.rand(5, 3)
    print(x + y)
    print(torch.add(x, y))
    result = torch.empty(5, 3)
    torch.add(x, y, out=result)
    print(result)
    # y에 오버라이딩
    y.add_(x)
    print(y)



