import pandas as pd
# ttest_ind : 독립 2- 표본 ttest
# ttest_rel : 1- 표본 ttest 
from scipy.stats import ttest_ind
# from scipy.stats import ttest_rel

# chain01_datawig
# chain03_datawig
# up_chain03_datawig

# CSV 파일 읽기
file_path = './res/chain1+chain3/29_echocardiogram_chain1+3_iteration_rmse_stats.csv'  # 파일 경로를 실제 파일 경로로 대체
data  = pd.read_csv(file_path)

# 'Accuracy' 열을 숫자로 변환
data['RMSE'] = data['RMSE'].str.extract('(\d+\.\d+)').astype(float)

# method 목록 추출
methods = data['method'].unique()

# 1등 성능으로 가정할 method 선택
top_method = 'up_chain03_datawig'

# 1등 성능 method와 나머지 method들 간의 t-test 수행
for method in data['method'].unique():
    if method != top_method:
        # 1등 성능 method와 다른 method 간의 데이터 추출
        data_top = data[data['method'] == top_method]['RMSE']
        data_current = data[data['method'] == method]['RMSE']

        # t-test 수행
        t_statistic, p_value = ttest_ind(data_top, data_current)
        # t_statistic, p_value = ttest_rel(data_top, data_current)

        # 결과 출력
        print(f"{top_method}와 {method} 간의 t-test 결과:")
        print(f"  T-통계량: {t_statistic}")
        print(f"  P-값: {p_value}")
        print("")

        # P-값을 통해 유의수준을 설정하고 (예: 0.05), 결과를 해석할 수 있습니다.
        if p_value < 0.05:
            print(f"  {top_method}이(가) {method}보다 성능이 우수합니다.")
        else:
            print(f"  {top_method}과(와) {method} 사이에는 유의한 성능 차이가 없습니다.")
        print("\n" + "="*50 + "\n")


# # 각 method에 대한 t-test 수행
# for i in range(len(methods)):
#     for j in range(i+1, len(methods)):
#         method1 = methods[i]
#         method2 = methods[j]

#         # 각 method에 대한 데이터 추출
#         data_method1 = data[data['method'] == method1]['Accuracy']
#         data_method2 = data[data['method'] == method2]['Accuracy']

#         # t-test 수행
#         t_statistic, p_value = ttest_rel(data_method1, data_method2)

#         # 결과 출력
#         print(f"{method1}와 {method2} 간의 t-test 결과:")
#         print(f"  T-통계량: {t_statistic}")
#         print(f"  P-값: {p_value}")
#         print("")

#         # P-값을 통해 유의수준을 설정하고 (예: 0.05), 결과를 해석할 수 있습니다.
#         if p_value < 0.05:
#             print(f"  {method1}과(와) {method2} 사이에는 유의한 차이가 있습니다.")
#         else:
#             print(f"  {method1}과(와) {method2} 사이에는 유의한 차이가 없습니다.")
#         print("\n" + "="*50 + "\n")