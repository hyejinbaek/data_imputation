import pandas as pd
import numpy as np
import os

# prepro_data = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing/2_heart.csv'
# prepro_data = pd.read_csv(prepro_data)

# print(prepro_data)

# 폴더 내의 모든 CSV 파일 경로를 가져오는 함수
def get_csv_files(folder_path):
    csv_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            csv_files.append(os.path.join(folder_path, file))
    return csv_files

# 각 CSV 파일의 데이터프레임의 shape를 출력하는 함수
def print_dataframe_shapes(csv_files):
    for file in csv_files:
        df = pd.read_csv(file)
        print(f"File: {file}, Shape: {df.shape}")


folder_path = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing'

# 폴더 내의 CSV 파일 경로들을 가져옴
csv_files = get_csv_files(folder_path)

# 각 CSV 파일의 데이터프레임의 shape를 출력
print_dataframe_shapes(csv_files)