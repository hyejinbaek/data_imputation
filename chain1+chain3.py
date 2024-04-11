import pandas as pd
import os

# 결과를 저장할 디렉토리 경로
output_directory = '/userHome/userhome2/hyejin/test/res/chain1+chain3'

# 파일이 있는 디렉토리 경로
input_directory = '/userHome/userhome2/hyejin/test/res/'

# 모든 파일의 경로를 저장할 딕셔너리
file_dict = {}

# 파일 경로 수집
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        # 파일명에서 숫자 부분 추출
        number = filename.split('_')[0]
        file_path = os.path.join(input_directory, filename)
        
        # 딕셔너리에 파일 경로 추가
        if number in file_dict:
            file_dict[number].append(file_path)
        else:
            file_dict[number] = [file_path]

# 같은 번호끼리 파일을 합치기
for number, file_paths in file_dict.items():
    # 파일들을 읽어서 DataFrame으로 저장
    dfs = [pd.read_csv(file_path) for file_path in file_paths]
    
    # DataFrame 합치기
    combined_df = pd.concat(dfs, axis=1)
    
    # 결과 저장
    output_file_path = os.path.join(output_directory, f'combined_{number}.csv')
    combined_df.to_csv(output_file_path, index=False)
