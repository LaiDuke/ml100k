import pandas as pd
from surprise import *
from surprise import Dataset
from surprise import Reader, accuracy
import time

df_train = pd.read_csv(r'training.csv', sep='\t')
df_train.drop(df_train.columns[[0]], axis=1, inplace=True)
reader = Reader(rating_scale=(0.5, 5))
train_ = Dataset.load_from_df(df_train[['userId','movieId','rating']], reader)
df_test = pd.read_csv(r'testing.csv', sep='\t')
df_test.drop(df_test.columns[[0]], axis=1, inplace=True)
test_ = Dataset.load_from_df(df_test[['userId','movieId','rating']], reader)
trainset = train_.build_full_trainset()
testset = test_.build_full_trainset().build_testset()


#SVD
start = time.time()
algo_SVD = SVD()
algo_SVD.fit(trainset)
predictions_SVD = algo_SVD.test(testset)
print(pd.DataFrame(predictions_SVD))
end = time.time()
print('Thoi gian thuc hien SVD: ', (end - start))
print(accuracy.rmse(predictions_SVD))

#normal
start = time.time()
algo_normal_predictor = NormalPredictor()
algo_normal_predictor.fit(trainset)
predictions_normal = algo_normal_predictor.test(testset)
print(pd.DataFrame(predictions_normal))
end = time.time()
print('Thoi gian thuc hien normal: ', (end - start))
print(accuracy.rmse(predictions_normal))

#BaselineOnly
start = time.time()
algo_Baseline_Only = BaselineOnly()
algo_Baseline_Only.fit(trainset)
predictions_Baseline_Only = algo_Baseline_Only.test(testset)
print(pd.DataFrame(predictions_Baseline_Only))
end = time.time()
print('Thoi gian thuc hien BaselineOnly: ', (end - start))
print(accuracy.rmse(predictions_Baseline_Only))