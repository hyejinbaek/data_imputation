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
result_csv_path = '/userHome/userhome2/hyejin/test/res/chain1/11_iris_datawig_method_res.csv'

# 결과를 저장할 리스트 초기화
results = []


def main():

    prepro_data = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing/11_iris.csv'
    prepro_data = pd.read_csv(prepro_data)

    data_pth = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/missing/11_iris.csv'
    df_data = pd.read_csv(data_pth)
    train_col = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    data_with_missing = df_data

    # 반복 횟수 설정
    num_iterations = 30

    for i in range(num_iterations):
        print(f"Iteration {i+1}")

        prev_imputed = None
        iteration_rmse_list = []
        for col in train_col:
            input_columns = train_col[:train_col.index(col)+1]
            print("==== input_columns ===", input_columns)
            imputer = SimpleImputer(
                input_columns=input_columns,
                output_column=col,
                output_path=f'./imputer_model/11_iris/imputer_model_{col}',
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
                'Dataset': '11_iris',
                'method': 'chain01_datawig',
                'Iteration': i + 1,
                'Column': col,
                f'{i+1}th Iteration RMSE': rmse
            }
            results.append(result)

    print("==========================================")
    print("=== RMSE result : {:.4f} ± {:.4f}".format(np.mean(iteration_rmse_list), np.std(iteration_rmse_list)))
    print("==========================================")

    # 결과를 DataFrame으로 변환하여 CSV 파일에 추가로 저장
    results_df = pd.DataFrame(results)
    if os.path.exists(result_csv_path):
        results_df.to_csv(result_csv_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(result_csv_path, index=False)

    print("Results saved to:", result_csv_path)

if __name__ == "__main__":
    main()