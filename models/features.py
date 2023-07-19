import pickle
from tqdm import tqdm
from scipy import stats
import copy
from functools import reduce
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def get_df_x(list_tmp):

    df_x = reduce(lambda left, right: pd.merge(left=left, right=right, left_index=True, right_index=True, how="left"),
                  list_tmp)

    #  drop columns , 너무 적은 데이터 제거
    df_tmp = pd.DataFrame()
    for col_nm in df_x.columns:
        nan_len = len(df_x[df_x[col_nm].isna()])
        df_tmp = df_tmp.append({"col_nm": col_nm, "count": nan_len}, ignore_index=True)

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

        with open(r'D:\MyProject\MyData\MacroData\MacroDataInfo.pickle', 'rb') as fr:
            self.dict_macro_data_info = pickle.load(fr)

        self.df_macro_data = pd.DataFrame()
        self.__set_macro_data()

    def __set_macro_data(self):

        # 1. 매크로 데이터 형 변환
        window_size = 12 * 3  # z_score 산출 window size

        for macro_type in self.dict_macro_data.keys():

            for macro_type_sub in self.dict_macro_data[macro_type].keys():

                for key_nm, df_macro in tqdm(self.dict_macro_data[macro_type][macro_type_sub].items()):

                    freq = self.dict_macro_data_info[macro_type][macro_type_sub][key_nm]["freq"]

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

                    self.dict_macro_data[macro_type][macro_type_sub][key_nm] = df_macro_data

        # 2. 자료구조 변경 (dict -> DataFrame)
        list_tmp = []
        for macro_type in self.dict_macro_data.keys():
            for macro_type_sub in self.dict_macro_data[macro_type].keys():
                for key_nm, df_macro in tqdm(self.dict_macro_data[macro_type][macro_type_sub].items()):
                    df = copy.deepcopy(df_macro)
                    df = df.rename(columns={"z_score": key_nm})

                    list_tmp.append(df[df.columns[-1:]])

        self.df_macro_data = get_df_x(list_tmp)
        print("created features ")

