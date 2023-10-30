import pickle
from tqdm import tqdm
from scipy import stats
import copy
from functools import reduce
import pandas as pd
import numpy as np
import warnings
import db
from datetime import datetime

warnings.filterwarnings('ignore')


def get_df_x(list_tmp):

    df_x = reduce(lambda left, right: pd.merge(left=left, right=right, left_index=True, right_index=True, how="left"),
                  list_tmp)

    #  drop columns , 너무 적은 데이터 제거
    stack_ = []
    for col_nm in df_x.columns:
        nan_len = len(df_x[df_x[col_nm].isna()])
        stack_.append(pd.DataFrame({"col_nm": [col_nm], "count": [nan_len]}))
    df_tmp = pd.concat(stack_)

    drop_col_list = df_tmp[df_tmp["count"] > 350]["col_nm"].to_list()
    df_x = df_x.drop(columns=drop_col_list)
    df_x = df_x.fillna(method="ffill")  # 최근 업데이트가 없는 경우 ffill
    df_x = df_x.dropna(axis=0)

    df_x = df_x.resample("1M").last()
    df_x = df_x.replace([np.inf, -np.inf], np.nan)

    return df_x


class Features:

    def __init__(self):

        with open(r'D:\MyProject\MyData\MacroData\MacroData.pickle', 'rb') as fr:
            self.dict_macro_data = pickle.load(fr)

        q = 'SELECT * FROM financial_data.macro_info'
        self.df_macro_info = pd.read_sql_query(q, db.conn)
        self.df_macro_info = self.df_macro_info.drop_duplicates("ticker")  # 임시

        self.df_macro_data = pd.DataFrame()
        self.__set_macro_data()

    def __set_macro_data(self):

        # 1. 매크로 데이터 형 변환
        window_size = 12 * 3  # z_score 산출 window size

        for i, rows in tqdm(self.df_macro_info.iterrows(), total=len(self.df_macro_info)):

            ticker = rows["ticker"]
            freq = rows["freq"]

            df_macro = self.dict_macro_data[ticker]

            apply_col = "val"
            if freq == "d":
                apply_col = "val"
            elif freq == "m":
                apply_col = "pct_chg"
            elif freq == "q":
                apply_col = "pct_chg"
            elif freq == "y":
                apply_col = "pct_chg"

            df_macro_data = df_macro.resample("1M").last().fillna(method='ffill')
            df_macro_data = df_macro_data[~df_macro_data[apply_col].isna()]
            df_macro_data["z_score"] = df_macro_data[apply_col].rolling(window_size).apply(lambda x: stats.zscore(x)[-1])

            self.dict_macro_data[ticker] = df_macro_data

        # 2. 자료구조 변경 (dict -> DataFrame)
        list_tmp = []
        for i, rows in tqdm(self.df_macro_info.iterrows(), total=len(self.df_macro_info)):

            ticker = rows["ticker"]
            df = copy.deepcopy(self.dict_macro_data[ticker])

            # 업데이트 일자가 3달 초과한 데이터는 pass
            if len(df) == 0:
                continue

            # 업데이트 일자가 3달 초과한 데이터는 pass
            if (datetime.today() - df.index[-1]).days > 90:
                continue

            df = df.rename(columns={"z_score": ticker})

            list_tmp.append(df[df.columns[-1:]])

        self.df_macro_data = get_df_x(list_tmp)
        print("created features ")

