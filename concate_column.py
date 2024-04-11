# 실험 결과 csv파일을 반복 횟수 만큼의 RMSE 계산하는 코드

import pandas as pd
import numpy as np

# CSV 파일을 읽어옵니다.
df = pd.read_csv('/userHome/userhome2/hyejin/test/res/chain3/up_33_forty_datawig_method_res.csv')

# Iteration 컬럼을 기준으로 그룹화합니다.
grouped = df.groupby('Iteration')

# 각 Iteration 별로 평균과 표준 편차를 계산합니다.
means = grouped.mean()
stds = grouped.std()

# 각 Iteration에 대한 평균과 표준 편차를 리스트에 저장합니다.
rmse_means = means.mean(axis=1).tolist()
rmse_stds = stds.mean(axis=1).tolist()

# 새로운 DataFrame을 만듭니다.
result_df = pd.DataFrame({
    'Dataset': ['33_forty'] * len(rmse_means),
    'method': ['up_chain03_datawig'] * len(rmse_means),
    'Experiment': means.index,
    'RMSE': ["{:.4f} ± {:.4f}".format(mean, std) for mean, std in zip(rmse_means, rmse_stds)]
})

# 결과를 CSV 파일로 저장합니다.
result_df.to_csv('/userHome/userhome2/hyejin/test/res/chain3/iteration_rmse_stats/up_33_forty_chain3_iteration_rmse_stats.csv', index=False)
