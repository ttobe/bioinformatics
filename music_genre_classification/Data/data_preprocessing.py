# 데이터를 빠르게 처리하기 위해서 pandas를 사용했습니다.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 매개변수: one hot encoding 여부, sh = 셔플사용 여부
def data_preprocessing(one_hot_encoding = True, sh = True):
    # 같은 위에 있는 데이터 불러오기
    df = pd.read_csv('Data/music_genre.csv')
    # 안쓸 column을 제외하기
    df = df.drop(columns= ['instance_id','artist_name','track_name', 'obtained_date'])

    # tempo columns에 ?가 있는 data를 버리기
    col = df.columns 
    if 'tempo' in col:
        idx = df[df['tempo'] == '?'].index
        df = df.drop(idx) 
    # 이제 tempo에서 ?인 값이 사라졌으니 float으로 데이터 타입을 바꿔주기
    df['tempo'] = df['tempo'].astype("float")
    
    # mode를 숫자로 바꾸기
    # sklearn의 LabelEncoder을 사용했습니다.
    key_encoder = LabelEncoder()
    df["key"] = key_encoder.fit_transform(df["key"])
    # mode key 같이 적용하기 위한 코드
    # ex) C Major과 C Minor는 12차이
    df = df.replace({'Minor': 0, 'Major' : 12})
    df['key'] = df['mode'] + df['key']
    # 합쳐진 mode column은 버리기
    df = df.drop(columns= ['mode'])
    
    # 학습에 제외할 label 배열
    drop_col = ['Anime','Country', 'Jazz']
    # 제외할 label을 가진 data를 버리기
    for col in drop_col:
        idx = df[df['music_genre'] == col].index
        df = df.drop(idx)
    # 데이터 레이블에 대한 int로 인코딩
    # Alternatvie와 Rock을 같은 장르로, Hip-Hop과 Rap을 같은 장르로 판별
    df = df.replace({'Electronic' : 0, 'Alternative' : 1, 'Rock': 1, 'Rap' : 2, 'Hip-Hop': 2, 'Blues' : 3, 'Classical' : 4})

    # 처리 후 데이터 plot
    # df.hist(figsize=(10, 9))
    # plt.tight_layout()
    # plt.show()

    # 정규화 min max로 한거 정규화 
    df = (df - df.min())/(df.max() - df.min())
 
    # 정답 레이블은 원상복구를 합니다.
    df['music_genre'] = df['music_genre'] * 4
    
    # 데이터 정규화 이후에 floating
    # df.hist(figsize=(10, 9))
    # plt.tight_layout()
    # plt.show()
    
    # 섞여있는 데이터를 label을 기준으로 정렬하기
    df = df.sort_values(by=['music_genre'])
    # label당 데이터가 몇개씩 있는지 확인하기
    data_count = df['music_genre'].value_counts() 
    
    # 편하게 사용하기 위해서 numpy로 변환하기
    data = df.to_numpy()
    
    # input feature과 정답 t를 나눔
    x = data[:,:data.shape[1]-1]
    t = data[:,data.shape[1]-1:]
    # 정답의 타입변환
    t = t.astype(np.int64)
    
    # 변수지정
    # train_start = 레이블 당 학습 데이터가 시작하는 index
    # test_start = 레이블 당 테스트 데이터가 시작하는 index = 학습 데이터가 끝나는 index
    # test_last = 레이블 당 테스트 데이터가 끝나는 index
    # 처음에는 선언하기위해 따로 빼줘서 계산
    test_last = 0
    # train data의 비율 
    train_rate = 0.80
    # 0번째 label 트레인 데이터 개수
    train_num = int(data_count[0] * train_rate)
    # train data 시작지점
    train_start = test_last
    # test data 시작지점 = train data 끝나는 지점
    test_start = test_last + train_num
    # test data 끝나는 지점
    test_last += data_count[0]
    # 선언 겸 첫번째 저장하기
    x_train=x[train_start:test_start]
    t_train=t[train_start:test_start]
    x_test=x[test_start:test_last]
    t_test=t[test_start:test_last]
    
    for k in range(1, 5):
        # 변수지정
        # train_start = 레이블 당 학습 데이터가 시작하는 index
        # test_start = 레이블 당 테스트 데이터가 시작하는 index = 학습 데이터가 끝나는 index
        # test_last = 레이블 당 테스트 데이터가 끝나는 index
        # k번째 label 트레인 데이터 개수
        train_num = int(data_count[k] * train_rate)
        # train data 시작지점
        train_start = test_last
        # test data 시작지점 = train data 끝나는 지점
        test_start = test_last + train_num
        # test data 끝나는 지점
        test_last += data_count[k]
        # 더하기 합치기
        x_train = np.concatenate((x_train,x[train_start:test_start]),axis=0)
        x_test = np.concatenate((x_test,x[test_start:test_last]),axis=0)
        t_train = np.concatenate((t_train,t[train_start :test_start]),axis=0)
        t_test = np.concatenate((t_test,t[test_start:test_last]),axis=0)
    
    # 정답 레이블의 1차원으로 차원 낮추기
    t_train = np.ravel(t_train)
    t_test = np.ravel(t_test)

    # one hot encoding을 적용시키기
    if one_hot_encoding == one_hot_encoding:
        num = np.unique(t_train, axis=0)
        num = num.shape[0]
        t_train = np.eye(num)[t_train]
        t_test = np.eye(num)[t_test] 
    print(x_train.shape, x_test.shape, t_train.shape, t_test.shape)

    return (x_train, t_train), (x_test, t_test)

