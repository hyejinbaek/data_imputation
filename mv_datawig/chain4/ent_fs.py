import numpy as np
import pandas as pd
from scipy.io import loadmat


def entropy_fs(X):
    print(" === X === ", X)
    fs_idx = np.zeros(X.shape[1])
    fs_score = np.zeros(X.shape[1])
    ent_map = np.zeros((X.shape[1], X.shape[1]))

    for i in range(0, X.shape[1]):
        for j in range(i, X.shape[1]):
            if i != j:
                ent_map[i, j] = joint_entropy(X[:, i], X[:, j])

    ent_map = ent_map + ent_map.T

    # 수식
    for i in range(0, X.shape[1]):
        fs_temp = 0
        for j in range(X.shape[1]):
            if j != i:
                fs_temp += ent_map[i, j]

        fs_score[i] = fs_temp

    fs_idx = np.argsort(fs_score)
    return fs_idx, fs_score


def joint_entropy(f_i, f_j):
    # f_i와 f_j의 결합된 데이터 생성
    joint_data = np.vstack((f_i, f_j)).T

    # 고유한 행 조합 및 각 조합의 출현 빈도 계산
    unique_joint_combinations, counts = np.unique(
        joint_data, axis=0, return_counts=True
    )

    # 전체 데이터 포인트의 수
    total_counts = np.sum(counts)

    # 엔트로피 계산
    entropy = -np.sum((counts / total_counts) * np.log2(counts / total_counts))

    return entropy


def main():
    # load ba.csv
    # df = pd.read_csv("colon.csv")
    # X = df.values

    # load ba.mat
    dat = loadmat("ba.mat")
    print(dat)
    X = dat["fea"]

    # 엔트로피 기반 피처 선택
    fs_idx, fs_score = entropy_fs(X)

    print("Feature Index: ", fs_idx)
    print("Feature Score: ", fs_score)


if __name__ == "__main__":
    main()
