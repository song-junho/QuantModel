import models.features

def main():

    pass

if __name__ == "__main__":

    # 1. Feature 로 사용할 매크로 변수 초기화
    c_features = models.features.Features()
    df_macro_features = c_features.df_macro_data

    # 2. 모델 학습
    