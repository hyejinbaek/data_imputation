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
result_csv_path = '/userHome/userhome2/hyejin/test/res/chain3/15_magic_datawig_method_res.csv'


# 결과를 저장할 리스트 초기화
results = []

# 등빈도 분할 후 entropy를 계산하는 함수
def equal_frequency_binning(column, num_bins=3):
    bins = pd.qcut(column, q=num_bins, duplicates='drop')
    value_counts = bins.value_counts(normalize=True)
    probabilities = value_counts.values
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# 주어진 데이터프레임의 각 컬럼에 대해 entropy를 계산하는 함수
def calculate_entropy_for_columns(dataframe, num_bins=10):
    entropies = {}
    for column in dataframe.columns:
        if dataframe[column].dtype in ['int64', 'float64']:
            entropy = equal_frequency_binning(dataframe[column], num_bins)
            entropies[column] = entropy
    sorted_entropies = dict(sorted(entropies.items(), key=lambda item: item[1]))
    return sorted_entropies

def main():

    prepro_data = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing/15_magic.csv'
    prepro_data = pd.read_csv(prepro_data)

    data_pth = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/missing/15_magic.csv'
    df_data = pd.read_csv(data_pth)
    train_col = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']

    data_with_missing = df_data

    # 등빈도 분할 후 각 컬럼의 entropy 계산
    entropies = calculate_entropy_for_columns(data_with_missing[train_col])
    print("======= Entropy for each column:", entropies)

    # Entropy를 기준으로 컬럼 순서를 정렬
    sorted_columns = sorted(entropies, key=entropies.get)
    print("======= Sorted columns based on entropy:", sorted_columns)


    # 반복 횟수 설정
    num_iterations = 30

    rmse_list = []
    imputers = {}
    previous_imputed = {}

    for iteration in range(num_iterations):
        # Train set과 test set으로 분할
        train_data, test_data = train_test_split(data_with_missing, test_size=0.2, random_state=iteration)

        # 데이터 결측치 채우기
        for col in sorted_columns:  # entropy가 낮은 순서대로 처리
            input_columns = sorted_columns[:sorted_columns.index(col) + 1]  # 현재 컬럼까지의 모든 컬럼
            print("==== input_columns ===", input_columns)
            imputer = SimpleImputer(
                input_columns=input_columns,
                output_column=col,
                output_path=f'./imputer_model/15_magic/imputer_model_{col}',
                num_hash_buckets=2 ** 15,
                num_labels=100,
                tokens='chars',
                numeric_latent_dim=100,
                numeric_hidden_layers=1,
                is_explainable=False)
            
            imputer.fit(train_df=train_data, num_epochs=5)
            imputers[col] = imputer

        # Impute missing values for each column in train_data
        train_imputed_data = {}
        for col, imputer in imputers.items():
            if previous_imputed.get(col) is not None:  # 이전에 imputed된 데이터가 있는 경우에만 처리합니다.
                previous_imputed[col] = previous_imputed[col].rename(col + '_previous_imputed')
                train_data = pd.concat([train_data, previous_imputed[col]], axis=1)
            predictions = imputer.predict(train_data)
            train_imputed_data[col] = predictions[col + '_imputed']  # '_imputed' is added by datawig

        previous_imputed = train_imputed_data.copy()
        # print(" === previous_imputed === ",previous_imputed)

        # Impute missing values for each column in test_data
        test_imputed_data = {}
        for col, imputer in imputers.items():
            predictions = imputer.predict(test_data)
            test_imputed_data[col] = predictions[col + '_imputed']  # '_imputed' is added by datawig

        # Create a DataFrame with imputed values for test set
        test_imputed_df = pd.DataFrame(test_imputed_data)

        # 결측치 생성 전의 데이터를 동일하게 train/test로 나누어서 저장
        original_data_train, original_data_test = train_test_split(prepro_data, test_size=0.2, random_state=iteration)
        original_data_test = original_data_test.drop(columns=['class'])

        # Min-Max Scaling 수행
        scaler = MinMaxScaler(feature_range=(-1, 1))  # imputed_test_data와 동일한 범위로 조정
        original_x_test_scaled = scaler.fit_transform(original_data_test)
        test_X_scaled = scaler.fit_transform(test_imputed_df)

        # RMSE 계산
        rmse = sqrt(mean_squared_error(original_x_test_scaled, test_X_scaled))
        print("==========================================")
        print(str(iteration + 1) + "th Ensemble Imputation rmse: ", rmse)
        print("==========================================")
        rmse_list.append(rmse)

        # 결과를 딕셔너리로 저장
        result = {
            'Dataset' : '15_magic',
            'method' : 'chain03_datawig',
            'Experiment': iteration + 1,
            'Entropy for each column': entropies,
            'Sorted columns based on entropy': ", ".join(sorted_columns),
            'RMSE': "{:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list))
        }
        results.append(result)

    print("==========================================")
    print("=== RMSE result : {:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)))
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