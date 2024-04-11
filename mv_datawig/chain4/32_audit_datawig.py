import warnings
warnings.filterwarnings("ignore")
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import sys
sys.path.append('/userHome/userhome2/hyejin/test/mv_datawig')
from simple_imputer import SimpleImputer

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/test/res/chain4/32_audit_datawig_method_res.csv'

# 결과를 저장할 리스트 초기화
results = []

# 등너비 분할 후 entropy를 계산하는 함수
def equal_width_binning(column, num_bins=5):
    # 최소값, 최대값 구하기
    min_val = column.min()
    max_val = column.max()
    # 등간격으로 나누기
    bins = np.linspace(min_val, max_val, num_bins + 1)
    # 범주로 변환
    digitized = np.digitize(column, bins)
    return digitized


# 수정된 entropy_fs 함수
def entropy_fs(X):
    n_features = X.shape[1]
    fs_score = np.zeros(n_features)
    ent_map = np.zeros((n_features, n_features))

    # 피처 간의 상호 엔트로피 계산
    for i in range(n_features):
        for j in range(i + 1, n_features):
            ent_map[i, j] = joint_entropy(X[:, i], X[:, j])

    # 피처 간의 상호 엔트로피를 대칭행렬로 만듦
    ent_map += ent_map.T

    # 각 피처의 중요도를 계산
    for i in range(n_features):
        fs_score[i] = np.sum(ent_map[i, :])

    # 중요도 순으로 피처 인덱스 정렬
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

    prepro_data = '/userHome/userhome2/hyejin/test/00_dataset/preprocessing/32_audit.csv'
    prepro_data = pd.read_csv(prepro_data)

    data_pth = '/userHome/userhome2/hyejin/test/00_dataset/missing/32_audit.csv'
    df_data = pd.read_csv(data_pth)
    col_data = df_data.columns
    train_col = list(col_data)
    train_col.remove('class')
    
    data_with_missing = df_data

    # 피처 선택 수행
    selected_features = entropy_fs(data_with_missing[train_col])
    print("Selected Features:", selected_features)

    # 반복 횟수 설정
    num_iterations = 30

    for i in range(num_iterations):
        print(f"Iteration {i+1}")

        prev_imputed = None
        iteration_rmse_list = []
        for col in selected_features:
            input_columns = selected_features[:selected_features.index(col) + 1]
            print("==== input_columns ===", input_columns)
            imputer = SimpleImputer(
                input_columns=input_columns,
                output_column=col,
                output_path=f'./imputer_model/32_audit/imputer_model_{col}',
                num_hash_buckets=2 ** 15,
                num_labels=100,
                tokens='chars',
                numeric_latent_dim=100,
                numeric_hidden_layers=1,
                is_explainable=False)

            # Train set과 test set으로 분할
            train_data, test_data = train_test_split(data_with_missing, test_size=0.2, random_state=i)

            # 데이터 결측치 채우기
            imputer.fit(train_df=train_data, num_epochs=5)

            # Impute missing values for each column in train_data
            train_imputed_data = {}
            if prev_imputed is not None:
                for prev_col in prev_imputed.columns:
                    train_data.loc[:, prev_col + '_imputed'] = prev_imputed[prev_col]
            train_imputed_data[col + '_imputed'] = imputer.predict(train_data)[col + '_imputed']
            prev_imputed = pd.DataFrame(train_imputed_data)

            # Impute missing values for each column in test_data
            test_imputed_data = {}
            if prev_imputed is not None:
                for prev_col in prev_imputed.columns:
                    test_data.loc[:, prev_col + '_imputed'] = prev_imputed[prev_col]
            test_imputed_data[col + '_imputed'] = imputer.predict(test_data)[col + '_imputed']


            # Create a DataFrame with imputed values for test set
            test_imputed_df = pd.DataFrame(test_imputed_data)

            # 결측치 생성 전의 데이터를 동일하게 train/test로 나누어서 저장
            original_data_train, original_data_test = train_test_split(prepro_data, test_size=0.2, random_state=i)
            original_data_test = original_data_test.drop(columns=['class'])

            # Min-Max Scaling 수행
            scaler = MinMaxScaler(feature_range=(-1, 1))  # imputed_test_data와 동일한 범위로 조정
            original_x_test_scaled = scaler.fit_transform(original_data_test[col].values.reshape(-1, 1))
            test_X_scaled = scaler.fit_transform(test_imputed_df[col + '_imputed'].values.reshape(-1, 1))

            # RMSE 계산
            rmse = sqrt(mean_squared_error(original_x_test_scaled, test_X_scaled))
            print(":::::::::: ",col + " Column Imputation rmse: ", rmse, ":::::::::: ")

            # 각 반복에서의 컬럼별 RMSE 값을 저장
            iteration_rmse_list.append(rmse)

            # 컬럼별 RMSE 값을 저장
            result = {
                'Dataset': '32_audit',
                'method': 'chain04_datawig',
                'Iteration': i + 1,
                'Column': col,
                f'{i+1}th Iteration RMSE': rmse
            }
            results.append(result)

    print("==========================================")
    print("=== RMSE result : {:.4f} ± {:.4f}".format(np.mean(iteration_rmse_list), np.std(iteration_rmse_list)))
    print("==========================================")

    # # 결과를 DataFrame으로 변환하여 CSV 파일에 추가로 저장
    # results_df = pd.DataFrame(results)
    # if os.path.exists(result_csv_path):
    #     results_df.to_csv(result_csv_path, mode='a', header=False, index=False)
    # else:
    #     results_df.to_csv(result_csv_path, index=False)

    # print("Results saved to:", result_csv_path)

if __name__ == "__main__":
    main()