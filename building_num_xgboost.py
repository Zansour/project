# -*- coding: utf-8 -*-
import os
os.chdir("C:/Users/Cha/Documents/power_prediction")

# 필요 모듈 불러오기
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from numpy import nan as NA

import sys
import utils
import graph
import sktime
from sklearn.preprocessing import LabelEncoder
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils.plotting import plot_series
from xgboost import XGBRegressor
from scipy import interpolate

import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.5f}'.format
import seaborn as sns

from tqdm import tqdm
import importlib
importlib.reload(utils)
importlib.reload(graph)

# =============================================================================
# 원본 데이터 불러오기 및 데이터 병합
# =============================================================================
building_info = pd.read_csv('building_info.csv')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 컬럼명 영어로 변환
train_df = utils.KortoEng_column(train_df)
test_df = utils.KortoEng_column(test_df)
building_info = utils.KortoEng_building(building_info)

# [건물 유형] 영어로 변환
building_info = utils.KortoEng_type(building_info)

# [건물 번호] 기준으로 train/test + building_info 데이터프레임 병합
train_df = pd.merge(train_df, building_info, on='building_number', how='left')
test_df = pd.merge(test_df, building_info, on='building_number', how='left')

# =============================================================================
# 결측치 처리
# - 결측치 비율이 높은 컬럼은 삭제하고, 그렇지 않은 컬럼은 선형 보간.
# =============================================================================
# [태양광용량], [ESS저장용량], [PCS용량] 컬럼 삭제
train_df['solar_power_capacity'].value_counts().head(5) / len(train_df) * 100
train_df['ess_capacity'].value_counts().head(5) / len(train_df) * 100
train_df['pcs_capacity'].value_counts().head(5) / len(train_df) * 100

train_df = train_df.drop(['solar_power_capacity', 'ess_capacity', 'pcs_capacity'], axis=1)
test_df = test_df.drop(['solar_power_capacity', 'ess_capacity', 'pcs_capacity'], axis=1)

# [강수량] 결측치 0으로 대치
sum(train_df['rainfall'].isnull()) / len(train_df) * 100  # train의 강수량 - 전체의 78%가 결측치

train_df['rainfall'].fillna(0.0, inplace=True)

# [일조], [일사] 컬럼 삭제
sum(train_df['sunshine'].isnull()) / len(train_df) * 100           # 전체의 36%
sum(train_df['solar_radiation'].isnull()) / len(train_df) * 100    # 전체의 43%

train_df = train_df.drop(['sunshine', 'solar_radiation'], axis=1)

# [풍속], [습도] 결측치 보간
train_df['windspeed'] = train_df['windspeed'].interpolate(method='linear')
train_df['humidity'] = train_df['humidity'].interpolate(method='linear')

# =============================================================================
# Feature Engineering
# =============================================================================
# 0. date time feature 생성 - [연도, 월, 일, 요일, 시간]
train_df = utils.datetime_parser(train_df)
test_df = utils.datetime_parser(test_df)

# 1. [건물 유형] 라벨 인코딩
le = LabelEncoder()
train_df['building_type'] = le.fit_transform(train_df['building_type'])
test_df['building_type'] = le.transform(test_df['building_type'])

# le.classes_

# 2. [건물 유형]별 특성을 반영할 수 있는 변수 생성
# 2-1) [건물 유형]별, [시간]별 최대 전력량
type_time_max = pd.pivot_table(train_df, index=['building_type', 'hour'], values='power_consumption', aggfunc = np.max).reset_index()
tqdm.pandas()
train_df['type_time_max'] = train_df.progress_apply(lambda x:type_time_max.loc[(type_time_max['building_type'] == x['building_type']) & (type_time_max['hour'] == x['hour']), 'power_consumption'].values[0], axis=1)
test_df['type_time_max'] = test_df.progress_apply(lambda x:type_time_max.loc[(type_time_max['building_type'] == x['building_type']) & (type_time_max['hour'] == x['hour']), 'power_consumption'].values[0], axis=1)

# 2-2) [건물 유형]별, [요일]별 최대 전력량
type_wday_max = pd.pivot_table(train_df, index=['building_type', 'weekday'], values='power_consumption', aggfunc = np.max).reset_index()
tqdm.pandas()
train_df['type_wday_max'] = train_df.progress_apply(lambda x:type_wday_max.loc[(type_wday_max['building_type'] == x['building_type']) & (type_wday_max['weekday'] == x['weekday']), 'power_consumption'].values[0], axis=1)
test_df['type_wday_max'] = test_df.progress_apply(lambda x:type_wday_max.loc[(type_wday_max['building_type'] == x['building_type']) & (type_wday_max['weekday'] == x['weekday']), 'power_consumption'].values[0], axis=1)

# 2-3) [건물 유형]별, [전력 주사용 시간대]별 최대 전력량
# ( EDA ) - [건물 유형]별 [전력 주사용 시간대(working_hour)] labeling
mean_power_by_hour_building = train_df.groupby(['hour', 'building_type'])['power_consumption'].mean().reset_index()
pivot_df = mean_power_by_hour_building.pivot(index='hour', columns='building_type', values='power_consumption')

plt.figure(figsize=(15, 10))
sns.lineplot(data=pivot_df)
plt.title('Mean Power Consumption by Hour of Day and Building Type')
plt.xlabel('Hour of Day')
plt.ylabel('Mean Power Consumption')
plt.legend(title='Building Type')
plt.show()

# - labeling한 컬럼으로 [건물 유형]별, [전력 주사용 시간대]별 최대 전력량 변수 생성
train_df = utils.working_hour(train_df)
test_df = utils.working_hour(test_df)

type_whour_max = pd.pivot_table(train_df, index=['building_type', 'working_hour'], values='power_consumption', aggfunc = np.max).reset_index()
tqdm.pandas()
train_df['type_whour_max'] = train_df.progress_apply(lambda x:type_whour_max.loc[(type_whour_max['building_type'] == x['building_type']) & (type_whour_max['working_hour'] == x['working_hour']), 'power_consumption'].values[0], axis=1)

tqdm.pandas()
test_df['type_whour_max'] = test_df.progress_apply(lambda x:type_whour_max.loc[(type_whour_max['building_type'] == x['building_type']) & (type_whour_max['working_hour'] == x['working_hour']), 'power_consumption'].values[0], axis=1)

# 3. [불쾌지수]
train_df['THI'] = 9/5*train_df['temperature'] - 0.55*(1-train_df['humidity']/100)*(9/5*train_df['humidity']-26)+32
test_df['THI'] = 9/5*test_df['temperature'] - 0.55*(1-test_df['humidity']/100)*(9/5*test_df['humidity']-26)+32

# 4. [냉방도일]
train_df = utils.cdh_df(train_df)
test_df = utils.cdh_df(test_df)

# 5. 시간의 연속적 표현을 위한 [요일], [시간] 변수 가공
train_df['sin_time'] = np.sin(2*np.pi*train_df['hour']/24)
train_df['sin_wday'] = np.sin(2*np.pi*train_df['weekday']/6)

test_df['sin_time'] = np.sin(2*np.pi*test_df['hour']/24)
test_df['sin_wday'] = np.sin(2*np.pi*test_df['weekday']/6)

# 6. [공휴일] 변수 가공
# 6-1) 토요일, 일요일 공휴일로 설정
tqdm.pandas()
train_df['holiday'] = train_df.progress_apply(lambda x : 0 if x['day']<5 else 1, axis = 1)

tqdm.pandas()
test_df['holiday'] = test_df.progress_apply(lambda x : 0 if x['day']<5 else 1, axis = 1)

# 6-2) 국가 지정 공휴일
train_df.loc['2022-08-15'==train_df['date_time'], 'holiday'] = 1  # 광복절
train_df.loc['2022-06-06'==train_df['date_time'], 'holiday'] = 1  # 현충일

test_df.loc['2022-08-15'==test_df['date_time'], 'holiday'] = 1  # 광복절
test_df.loc['2022-06-06'==test_df['date_time'], 'holiday'] = 1  # 현충일

# 6-3) 할인마트 휴무일 공휴일 처리 
# ( EDA )
imsi = train_df[train_df.building_type == 4]
imsi = imsi.groupby(['building_number', 'month', 'day'])[['power_consumption']].sum()
imsi.reset_index(inplace=True)

imsi['month'] = [str(i) for i in imsi['month']]
imsi['day'] = [str(i).zfill(2) for i in imsi['day']]
imsi['date'] = imsi['month'] + '/' + imsi['day']

mean_power_by_date_building = imsi.groupby(['date', 'building_number'])['power_consumption'].mean().reset_index()
pivot_df = mean_power_by_date_building.pivot(index='date', columns='building_number', values='power_consumption')

x_positions = list(range(0, 85, 3))
x_labels = [imsi['date'][i] for i in x_positions]

for num in pivot_df.columns :
    temp = pivot_df[num]
    
    plt.figure(figsize=(15, 10))
    plt.xticks(x_positions, x_labels)
    sns.lineplot(data=temp)
    
    plt.title('Building Number : ' + str(num))
    plt.xlabel('Day')
    plt.ylabel('Mean Power Consumption')
    
# - 85번 건물 : 휴일 미처리 
# - 86번 건물 : 6월 10일, 6월 26일, 7월 10일, 7월 24일, 8월 10일 -> (test) 8월 28일 공휴일 처리 필요
# - 87번~92번 건물 : 6월 12일, 6월 26일, 7월 10일, 7월 24일, 8월 14일 -> (test) 8월 27일 공휴일 처리 필요

for num in tqdm(range(86,93)) :
    if num == 86 :
        holi = ['2022-06-10', '2022-06-26', '2022-07-10', '2022-07-24', '2022-08-10']
        train_df.loc[(train_df['date_time'].isin(holi)) & (num==train_df['building_number']), 'holiday'] = 1
    else :
        holi = ['2022-06-12', '2022-06-26', '2022-07-10', '2022-07-24', '2022-08-14']
        train_df.loc[(train_df['date_time'].isin(holi)) & (num==train_df['building_number']), 'holiday'] = 1

for num in tqdm(range(86,93)) :
        holi = ['2022-08-28']
        test_df.loc[(test_df['date_time'].isin(holi)) & (num==test_df['building_number']), 'holiday'] = 1
        
# => [건물 유형]별 [공휴일]별 최대 전력량
type_holi = pd.pivot_table(train_df, index=['building_type', 'holiday'], values='power_consumption', aggfunc = np.max).reset_index()
tqdm.pandas()
train_df['type_holi'] = train_df.progress_apply(lambda x:type_holi.loc[(type_holi['building_type'] == x['building_type']) & (type_holi['holiday'] == x['holiday']), 'power_consumption'].values[0], axis=1)

tqdm.pandas()
test_df['type_holi'] = test_df.progress_apply(lambda x:type_holi.loc[(type_holi['building_type'] == x['building_type']) & (type_holi['holiday'] == x['holiday']), 'power_consumption'].values[0], axis=1)

# 7. [3시간 동안의 최대 전력량]
# train_add = pd.read_csv('train_230821.csv')
# train_df = pd.concat([train_df, train_add['3h_max']], axis=1)

# test_add = pd.read_csv('test_230821.csv')
# test_df = pd.concat([test_df, test_add['3h_max']], axis=1)

df1 = pd.concat([train_df['building_number'], train_df['hour'], train_df['weekday'], train_df['power_consumption']], axis=1)

def three_hour_max_function(x, y, z):
    # 주말
    if z in [5, 6]:
        target_df = df1.loc[df1['weekday'] >= 5, :]
    # 평일
    else:
        target_df = df1.loc[df1['weekday'] < 5, :]

    if y == 23:
        hour_filter = [y - 1, y, 0]
    elif y == 0:
        hour_filter = [23, y, y + 1]
    else:
        hour_filter = [y - 1, y, y + 1]

    filtered_df = target_df.loc[target_df['hour'].isin(hour_filter) & (target_df['building_number'] == x), :]
    hour_max = filtered_df['power_consumption'].max()
    return hour_max

answer = list(map(three_hour_max_function, df1['building_number'], df1['hour'], df1['weekday']))
train_df['3h_max'] = answer


def find_max(x, y, z) :
    target_df = train_df.loc[(train_df['building_number'] == x) & (train_df['hour'] == y) & (train_df['weekday'] == z), '3h_max'].unique()
    three_hour_max = target_df[0]
    return three_hour_max

answer_1 = list(map(find_max, test_df['building_number'], test_df['hour'], test_df['weekday']))
answer_1[0] == train_df.loc[(train_df['building_number'] == 0) & (train_df['hour'] == 0) & (train_df['weekday'] == 3), '3h_max'].unique()
test_df['3h_max'] = answer_1

# 8. [냉방 면적 비율] (= 냉방 면적 / 전체 면적) 
# 8-1) 냉방 면적 결측치 대체
# - 65, 66, 68번 건물 (아파트) 냉방면적 0으로 건물 유형이 아파트인 건물들의 냉방면적 평균 적용
temp = train_df.copy()
apt_c_area = temp.loc[temp['building_type'] == 0, 'cooling_area'].unique().sum() / 5
train_df.loc[train_df['building_number'].isin([65,66,68]), 'cooling_area'] = apt_c_area
test_df.loc[test_df['building_number'].isin([65,66,68]), 'cooling_area'] = apt_c_area

# 8-2) 냉방면적 이상치 대체
# - 77, 80번 건물 냉방면적 각각 1, 239로 현저히 낮음. 건물 유형이 지식산업센터인 건물의 냉방면적 평균 적용
know_c_area = np.mean(sorted(temp.loc[temp['building_type'] == 7, 'cooling_area'].unique(), reverse=True)[:-2])
train_df.loc[train_df['building_number'].isin([77,80]), 'cooling_area'] = know_c_area
test_df.loc[test_df['building_number'].isin([77,80]), 'cooling_area'] = know_c_area

# 8-3) 냉방면적 비율
train_df['c_area_con'] = train_df['cooling_area'] / train_df['total_area'] * train_df['3h_max']
test_df['c_area_con'] = test_df['cooling_area'] / test_df['total_area'] * test_df['3h_max']

# 9. 온도차
temp_gap_list = []
for num in range(1, 101):
    temp = train_df[train_df['building_number'] == num] 
    temp_gap = temp['temperature'] - temp['temperature'].shift(1)
    temp_gap = temp_gap.bfill()
    temp_gap_list.append(temp_gap)

train_df['temp_gap'] = pd.concat(temp_gap_list)

temp_gap_list = []
for num in range(1, 101):
    temp = test_df[test_df['building_number'] == num] 
    temp_gap = temp['temperature'] - temp['temperature'].shift(1)
    temp_gap = temp_gap.bfill()
    temp_gap_list.append(temp_gap)

test_df['temp_gap'] = pd.concat(temp_gap_list)

# 10. 불필요한 변수 제거
train_df.drop(['hour', 'total_area', 'cooling_area', 'building_type'], axis = 1, inplace = True)
test_df.drop(['hour', 'total_area', 'cooling_area', 'building_type'], axis = 1, inplace = True)

# =============================================================================
# 모델링 
# =============================================================================
# SMAPE 함수 
def SMAPE(true, pred):
    return np.mean((np.abs(true-pred))/(np.abs(true) + np.abs(pred))) * 100

# 커스터마이즈 손실 함수 - 과소예측 보정
def weighted_mse(alpha = 1):
    def weighted_mse_fixed(label, pred):
        residual = (label - pred).astype("float")
        grad = np.where(residual>0, -2*alpha*residual, -2*residual)
        hess = np.where(residual>0, 2*alpha, 2.0)
        return grad, hess
    return weighted_mse_fixed


# 그리드 서치
from sklearn.model_selection import PredefinedSplit, GridSearchCV
df = pd.DataFrame(columns = ['n_estimators', 'eta', 'min_child_weight','max_depth', 'colsample_bytree', 'subsample'])
preds = np.array([])

grid = {'n_estimators' : [100], 'eta' : [0.1], 'min_child_weight' : np.arange(1, 8, 1), 
        'max_depth' : np.arange(3,9,1) , 'colsample_bytree' :np.arange(0.8, 1.0, 0.1), 
        'subsample' :np.arange(0.7, 1.0, 0.1)} # fix the n_estimators & eta(learning rate)

from sklearn.metrics import make_scorer
smape = make_scorer(SMAPE, greater_is_better = False)

for i in tqdm(np.arange(1, 101)):
    y = train_df.loc[train_df.building_number == i, 'power_consumption']
    x = train_df.loc[train_df.building_number == i, ].iloc[:, 2:].drop('power_consumption', axis=1)
    y_train, y_test, x_train, x_test = temporal_train_test_split(y = y, X = x, test_size = 168)
    

    pds = PredefinedSplit(np.append(-np.ones(len(x)-168), np.zeros(168)))
    gcv = GridSearchCV(estimator = XGBRegressor(seed = 0),
                       param_grid = grid, scoring = smape, cv = pds, refit = True, verbose = True)
    
    
    gcv.fit(x, y)
    best = gcv.best_estimator_
    params = gcv.best_params_
    print(params)
    pred = best.predict(x_test)
    building = 'building'+str(i)
    print(building + '|| SMAPE : {}'.format(SMAPE(y_test, pred)))
    preds = np.append(preds, pred)
    df = pd.concat([df, pd.DataFrame(params, index = [0])], axis = 0)

df.to_csv('xgb_hyperparameter.csv',index=False)
xgb_params = pd.read_csv('hyperparameter_xgb_final.csv')

# 가중치가 100일 때 iteration값 구하기
scores = []   # smape 값을 저장할 list
best_it = []  # best iteration을 저장할 list
for i in tqdm(range(100)):
    y = train_df.loc[train_df['building_number'] == i+1, 'power_consumption']
    x = train_df.loc[train_df['building_number'] == i+1, ].iloc[:,2:].drop('power_consumption', axis=1)
    y_train, y_valid, x_train, x_valid = temporal_train_test_split(y = y, X = x, test_size = 168)
    
    xgb_reg = XGBRegressor(n_estimators = 10000, eta = 0.1, min_child_weight = xgb_params.iloc[i, 2],
                            max_depth = xgb_params.iloc[i, 3], colsample_bytree = xgb_params.iloc[i, 4], 
                            subsample = xgb_params.iloc[i, 5], seed=0)
    
    xgb_reg.set_params(**{'objective':weighted_mse(100)}) # alpha = 100으로 고정

    xgb_reg.fit(x_train, y_train, eval_set=[(x_train, y_train), 
                                            (x_valid, y_valid)], early_stopping_rounds=300, verbose=False)
    y_pred = xgb_reg.predict(x_valid)
    pred = pd.Series(y_pred)   
    
    sm = SMAPE(y_valid, y_pred)
    scores.append(sm)
    best_it.append(xgb_reg.best_iteration) ## 실제 best iteration은 이 값에 +1 해주어야 함.


# 최적의 과소추정 보정 가중치 구하기
alpha_list = []
smape_list = []
for i in tqdm(range(100)):
    y = train_df.loc[train_df.building_number == i+1, 'power_consumption']
    x = train_df.loc[train_df.building_number == i+1, ].iloc[:, 2:].drop('power_consumption', axis=1)
    y_train, y_test, x_train, x_test = temporal_train_test_split(y = y, X = x, test_size = 168)
    xgb = XGBRegressor(seed = 0,
                      n_estimators = best_it[i], eta = 0.1, min_child_weight = xgb_params.iloc[i, 2],
                      max_depth = xgb_params.iloc[i, 3], colsample_bytree = xgb_params.iloc[i, 4], subsample = xgb_params.iloc[i, 5])
    
    xgb.fit(x_train, y_train)
    pred0 = xgb.predict(x_test)
    best_alpha = 0
    score0 = SMAPE(y_test,pred0)
    
    for j in [1, 3, 5, 7, 10, 25, 50, 75, 100]:
        xgb = XGBRegressor(seed = 0,
                      n_estimators = best_it[i], eta = 0.1, min_child_weight = xgb_params.iloc[i, 2],
                      max_depth = xgb_params.iloc[i, 3], colsample_bytree = xgb_params.iloc[i, 4], subsample = xgb_params.iloc[i, 5])
        xgb.set_params(**{'objective' : weighted_mse(j)})
    
        xgb.fit(x_train, y_train)
        pred1 = xgb.predict(x_test)
        score1 = SMAPE(y_test, pred1)
        if score1 < score0:
            best_alpha = j
            score0 = score1
    
    alpha_list.append(best_alpha)
    smape_list.append(score0)
    print("building {} || best score : {} || alpha : {}".format(i+1, score0, best_alpha))

no_df = pd.DataFrame({'score':smape_list})
plt.bar(np.arange(len(no_df))+1, no_df['score'])
plt.plot([1,100], [6, 6], color = 'red')

xgb_params['alpha'] = alpha_list
xgb_params['best_it'] = best_it
xgb_params.head()

xgb_params.to_csv('hyperparameter_xgb_final.csv', index=False)

# =============================================================================
# 최종 학습 및 predict(seed ensemble)
# =============================================================================
xgb_params = pd.read_csv('hyperparameter_xgb_final.csv')
xgb_params.head()

# Seed Ensemble
preds = np.array([]) 
for i in tqdm(range(100)):
    
    pred_df = pd.DataFrame()   # 시드별 예측값을 담을 data frame
    
    for seed in [0,1,2,3,4,5]: # 각 시드별 예측
        y_train = train_df.loc[train_df.building_number == i+1, 'power_consumption']
        x_train, x_test = train_df.loc[train_df.building_number == i+1, ].iloc[:, 2:].drop('power_consumption', axis=1), test_df.loc[test_df.building_number == i+1, ].iloc[:,2:]
        x_test = x_test[x_train.columns]
        
        xgb = XGBRegressor(seed = seed, n_estimators = xgb_params.iloc[i, 7], eta = 0.1, 
                           min_child_weight = xgb_params.iloc[i, 2], max_depth = xgb_params.iloc[i, 3], 
                           colsample_bytree=xgb_params.iloc[i, 4], subsample=xgb_params.iloc[i, 5])
    
        if xgb_params.iloc[i,6] != 0:  # 만약 alpha가 0이 아니면 weighted_mse 사용
            xgb.set_params(**{'objective':weighted_mse(xgb_params.iloc[i,6])})
        
        xgb.fit(x_train, y_train)
        y_pred = xgb.predict(x_test)
        pred_df.loc[:,seed] = y_pred   # 각 시드별 예측 담기
        
    pred = pred_df.mean(axis=1)       
    preds = np.append(preds, pred) 
    
submission = pd.read_csv('sample_submission.csv')
submission['answer'] = preds
submission.to_csv('submission_xgb.csv', index = False)
