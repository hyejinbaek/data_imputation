import pandas as pd
import numpy as np
from setproctitle import setproctitle
from simple_imputer import SimpleImputer
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import os
# CUDA 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 프로세스 제목 설정
setproctitle('hyejin')

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/res/chain1/4_wine_ensemble_method_res.csv'

# 결과를 저장할 리스트 초기화
results = []

def main():
    # Example usage
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    prepro_data = '/userHome/userhome2/hyejin/test/00_dataset/preprocessing/4_wine.csv'
    prepro_data = pd.read_csv(prepro_data)

    data_pth = '/userHome/userhome2/hyejin/test/00_dataset/missing/4_wine.csv'
    df_data = pd.read_csv(data_pth)
    train_col = ['Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium',
                        'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity',
                        'Hue', 'OD280%2FOD315_of_diluted_wines', 'Proline']
    data_with_missing = df_data

    # 반복 횟수 설정
    num_iterations = 30

    rmse_list = []
    imputers = {}

    for iteration in range(num_iterations):
        train_data, test_data = train_test_split(data_with_missing, test_size=0.2, random_state=1)

        df_train = pd.DataFrame(train_data)
        df_test = pd.DataFrame(test_data)

        # Set up imputer model
        for col in train_col:
            imputer = SimpleImputer(
                input_columns=train_col, # train_col = 13개
                output_column=col,
                output_path=f'./imputer_model/imputer_model_{col}',
                # output_path=f'./imputer_model/4_wine_{col}',
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
            predictions = imputer.predict(df_train)
            train_imputed_data[col] = predictions[col + '_imputed']  # '_imputed' is added by datawig

        # Create a DataFrame with imputed values for train set
        train_imputed_df = pd.DataFrame(train_imputed_data)
        print("--- train_imputed === ", train_imputed_df)

        # Impute missing values for each column in test_data
        test_imputed_data = {}
        for col, imputer in imputers.items():
            predictions = imputer.predict(df_test)
            test_imputed_data[col] = predictions[col + '_imputed']  # '_imputed' is added by datawig

        # Create a DataFrame with imputed values for test set
        test_imputed_df = pd.DataFrame(test_imputed_data)
        print(test_imputed_df)
    
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
        'Dataset' : '4_wine',
        'method' : 'datawig',
        'Experiment': iteration + 1,
        'RMSE': "{:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list))
    }
    results.append(result)

    print("==========================================")
    print("=== RMSE result : {:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)))
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