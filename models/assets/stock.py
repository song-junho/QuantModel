from abc import *
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


class Stock(metaclass=ABCMeta):

    def __init__(self, df_features):

        self.df_x = df_features
        self.dict_df_y = {}
        self.asset_type = None
        self.feature_type = None

        self.dict_model = {}
        self.df_model_score = pd.DataFrame()

    @abstractmethod
    def set_type(self):
        self.asset_type = None
        self.feature_type = None

    @abstractmethod
    def set_y(self):
        self.dict_df_y = {}

    def save_model(self, target_freq):

        with open(r'D:\MyProject\QuantModel\dict_model_{}_{}_{}.pickle'.format(self.asset_type, self.feature_type, target_freq), 'wb') as fw:
            pickle.dump(self.dict_model, fw)
        with open(r'D:\MyProject\QuantModel\df_model_score_{}_{}_{}.pickle'.format(self.asset_type, self.feature_type, target_freq), 'wb') as fw:
            pickle.dump(self.df_model_score, fw)

    def create_model(self, target_freq="3M"):

        list_model_theme = []

        for key_nm in tqdm(self.dict_df_y.keys()):

            df_y = self.dict_df_y[key_nm][[target_freq]]
            if len(df_y) == 0:
                continue

            # 타겟데이터 라벨링
            ## '0' : 많이 하락한 경우
            ## '1' : 평이한 경우
            ## '2' : 많이 오른 경우
            df_y["class"] = "1"
            df_y["class"] = df_y[target_freq].apply(lambda x: "2" if x > 0.10 else ("0" if x < -0.10 else "1"))
            df_y = df_y.shift(1)  # 타겟 값 한달뒤로 shift

            # class 0 값 개수 조절
            class_0 = df_y[df_y["class"] == "0"]
            class_1 = df_y[df_y["class"] == "1"]
            class_2 = df_y[df_y["class"] == "2"]

            avg_count = int(np.mean(len(class_1) + len(class_2)))

            # 라벨링 데이터 분포 표준화
            ## '1' 로 분류되는 데이터 분포가 일반적으로 많기 때문에 해당 데이터는 랜덤으로 제거한다
            if len(class_0) > (avg_count * 1.3):
                drop_count = len(class_0) - int(np.mean(len(class_1) + len(class_2)))
                np.random.seed(1004)
                drop_indices = np.random.choice(class_0.index, drop_count, replace=False)
                df_y = df_y.drop(drop_indices)
            else:
                pass

            # 타겟 데이터 업데이트가 끊긴 경우 pass
            if max(df_y.index) < max(self.df_x.index):
                print(key_nm)
                continue

            # 1. target 값 na인 경우 제거
            df_y = df_y[~df_y[target_freq].isna()][["class"]]

            list_time = set(self.df_x.index) & set(df_y.index)

            # 1. y 변인 일자 필터링
            df_y = df_y.loc[list_time]

            # 2. x 변인 일자 필터링
            df_x = self.df_x.loc[list_time]
            df_x = df_x.replace([np.nan], 0)

            # 3. train, test 데이터 분리
            X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, random_state=42)

            # 4. 모델 학습
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.fit_transform(y_test)

            best_model = xgb.XGBClassifier()
            best_model.fit(X_train, y_train)

            train_score = best_model.score(X_train, y_train)
            test_score = best_model.score(X_test, y_test)

            df = pd.DataFrame({"key_nm": [key_nm], "train_score": [train_score], "test_score": [test_score]})

            list_model_theme.append(df)
            self.dict_model[key_nm] = best_model

        # 모델 score 데이터 통합
        self.df_model_score = pd.concat(list_model_theme)

    def run(self, list_target_freq):

        self.set_type()
        self.set_y()

        for target_freq in list_target_freq:
            # 모델 생성
            self.create_model(target_freq)

            # 모델 저장
            self.save_model(target_freq)
            print("save model ", self.__class__.__name__, target_freq)


class KrxStock(Stock):

    def set_type(self):
        self.asset_type = 'stock'
        self.feature_type = 'krx'

    def set_y(self):
        with open(r'D:\MyProject\StockPrice\dict_stock_krx_chg_freq.pickle', 'rb') as fr:
            self.dict_df_y = pickle.load(fr)


class ThemeStock(Stock):

    def set_type(self):
        self.asset_type = 'stock'
        self.feature_type = 'theme'

    def set_y(self):
        with open(r'D:\MyProject\StockPrice\dict_stock_theme_chg_freq.pickle', 'rb') as fr:
            self.dict_df_y = pickle.load(fr)

