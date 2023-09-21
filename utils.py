# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
tqdm.pandas()

# 영어 변환 함수
def KortoEng_column(df : pd.DataFrame) :
    df = df.rename(columns={
    '건물번호': 'building_number',
    '일시': 'date_time',
    '기온(C)': 'temperature',
    '강수량(mm)': 'rainfall',
    '풍속(m/s)': 'windspeed',
    '습도(%)': 'humidity',
    '일조(hr)': 'sunshine',
    '일사(MJ/m2)': 'solar_radiation',
    '전력소비량(kWh)': 'power_consumption'
    })
    
    return df

def KortoEng_building(df : pd.DataFrame) :
    df = df.rename(columns={
    '건물번호': 'building_number',
    '건물유형': 'building_type',
    '연면적(m2)': 'total_area',
    '냉방면적(m2)': 'cooling_area',
    '태양광용량(kW)': 'solar_power_capacity',
    'ESS저장용량(kWh)': 'ess_capacity',
    'PCS용량(kW)': 'pcs_capacity'
    })
    
    return df
    
    
def KortoEng_type(df : pd.DataFrame) :
    translation_dict = {
    '건물기타': 'Other Buildings',
    '공공': 'Public',
    '대학교': 'University',
    '데이터센터': 'Data Center',
    '백화점및아울렛': 'Department Store and Outlet',
    '병원': 'Hospital',
    '상용': 'Commercial',
    '아파트': 'Apartment',
    '연구소': 'Research Institute',
    '지식산업센터': 'Knowledge Industry Center',
    '할인마트': 'Discount Mart',
    '호텔및리조트': 'Hotel and Resort'
    }
    
    df['building_type'] = df['building_type'].replace(translation_dict)
    
    return df
    
# 날짜 함수
def datetime_parser(df : pd.DataFrame) :
    df['date_time'] = pd.to_datetime(df['date_time'], format='%Y%m%d %H')

    # date time feature 생성
    df['hour'] = df['date_time'].dt.hour
    df['day'] = df['date_time'].dt.day
    df['month'] = df['date_time'].dt.month
    df['year'] = df['date_time'].dt.year
    df['weekday'] = df['date_time'].dt.weekday
    
    return df

# 냉방도일 
def cdh_df(df : pd.DataFrame) :
    
    def CDH(xs):
        ys = []
        for i in range(len(xs)):
            if i < 11:
                ys.append(np.sum(xs[:(i+1)]-26))
            else:
                ys.append(np.sum(xs[(i-11):(i+1)]-26))
        return np.array(ys)
    
    cdhs = np.array([])
    for num in range(1,101,1):
        temp = df[df['building_number'] == num]
        cdh = CDH(temp['temperature'].values)
        cdhs = np.concatenate([cdhs, cdh])
    df['CDH'] = cdhs
    
    return df

# working hour
def working_hour(df : pd.DataFrame) :
    
    # Apartment(0)
    condition = ((df['building_type']==0) & (df['hour']>=18) & (df['hour']<=23))
    df['working_hour'] = 0 
    df.loc[condition, 'working_hour'] = 1
    
    # Commercial(1)
    condition = ((df['building_type']==1) & (df['hour']>=9) & (df['hour']<=18))
    df.loc[condition, 'working_hour'] = 1
    
    # Data Center(2) ==> 없음
    # Department Store and Outlet(3)
    condition = ((df['building_type']==3) & (df['hour']>=10) & (df['hour']<=19))
    df.loc[condition, 'working_hour'] = 1
    
    # Discount Mart(4)
    condition = ((df['building_type']==4) & (df['hour']>=10) & (df['hour']<=19))
    df.loc[condition, 'working_hour'] = 1
    
    # Hospital(5)
    condition = ((df['building_type']==5) & (df['hour']>=8) & (df['hour']<=17))
    df.loc[condition, 'working_hour'] = 1
    
    # Hotel and Resort(6)
    condition = ((df['building_type']==6) & (df['hour']>=9) & (df['hour']<=19))
    df.loc[condition, 'working_hour'] = 1
    
    # Knowlege Industry Center(7)
    condition = ((df['building_type']==7) & (df['hour']>=9) & (df['hour']<=17))
    df.loc[condition, 'working_hour'] = 1
    
    # Other Buildings(8)
    condition = ((df['building_type']==8) & (df['hour']>=10) & (df['hour']<=17))
    df.loc[condition, 'working_hour'] = 1
    
    # Public(9)
    condition = ((df['building_type']==9) & (df['hour']>=8) & (df['hour']<=17))
    df.loc[condition, 'working_hour'] = 1
    
    # Research Institute(10)
    condition = ((df['building_type']==10) & (df['hour']>=9) & (df['hour']<=17))
    df.loc[condition, 'working_hour'] = 1
    
    # University(11)
    condition = ((df['building_type']==11) & (df['hour']>=9) & (df['hour']<=17))
    df.loc[condition, 'working_hour'] = 1
     
    return df
        
