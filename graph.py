# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from numpy import nan as NA

import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.5f}'.format
import seaborn as sns


def typeHourConsumption(df: pd.DataFrame, bd_type: str):
    func_df = df.loc[df['building_type'] == bd_type, :]
    pivot_df = func_df.groupby(['hour', 'building_number'])['power_consumption'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.lineplot(data=pivot_df)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    
    bd_type = bd_type.upper()
    plt.title(f'{bd_type}\n\n Mean Power Consumption by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Mean Power Consumption')
    plt.legend(title='Building Number', loc='upper left')
    
    return plt.show()
