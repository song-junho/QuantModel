import models.features
import models.assets.stock

if __name__ == "__main__":

    # 1. Feature 로 사용할 매크로 변수 초기화
    c_features = models.features.Features()
    df_macro_features = c_features.df_macro_data

    # 2. 모델 학습(Theme)
    c_model = models.assets.stock.ThemeStock(df_macro_features)
    c_model.run(["1M", "3M"])

    # 3. 모델 학습(Krx)
    c_model = models.assets.stock.KrxStock(df_macro_features)
    c_model.run(["1M", "3M"])
