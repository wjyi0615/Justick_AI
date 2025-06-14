import pandas as pd
import os
from datetime import datetime

# 작물 리스트
vegetables = ["tomato"]
rates = ["HIGH", "SPECIAL"]

# 날짜 범위 설정
start_date = datetime(2024, 4, 1)
end_date = datetime(2025, 5, 22)

for veg in vegetables:
    input_path = f'Data_250605/store/{veg}_separated.csv'
    output_path = f'Data_250605/store/{veg}_predict.csv'

    if not os.path.exists(input_path):
        print(f"{input_path} 파일이 없습니다.")
        continue

    df = pd.read_csv(input_path)

    # 날짜 컬럼 만들기
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # 조건 필터링: 날짜 범위 + 등급
    df_filtered = df[
        (df['date'] >= start_date) &
        (df['date'] <= end_date) &
        (df['rate'].isin(rates))
    ].copy()

    # 필요한 컬럼만
    final_df = df_filtered[['year', 'month', 'day', 'avg_price', 'rate']]
    final_df.sort_values(['rate', 'year', 'month', 'day'], inplace=True)

    os.makedirs('Data_250605/store', exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"{output_path} 저장 완료")
