import time

import pandas as pd
from surprise import *
from surprise import Dataset
from surprise import Reader, accuracy

df_train = pd.read_csv(r'training.csv', sep='\t')
df_train.drop(df_train.columns[[0]], axis=1, inplace=True)
reader = Reader(rating_scale=(0.5, 5))
train_ = Dataset.load_from_df(df_train[['userId', 'movieId', 'rating']], reader)
df_test = pd.read_csv(r'testing.csv', sep='\t')
df_test.drop(df_test.columns[[0]], axis=1, inplace=True)
test_ = Dataset.load_from_df(df_test[['userId', 'movieId', 'rating']], reader)
trainset = train_.build_full_trainset()
testset = test_.build_full_trainset().build_testset()

# SVD
start = time.time()
algo_SVD = SVD()
algo_SVD.fit(trainset)
predictions_SVD = algo_SVD.test(testset)
# print(pd.DataFrame(predictions_SVD))
end = time.time()
print('Thoi gian thuc hien SVD: ', (end - start))
print(accuracy.rmse(predictions_SVD))

# normal
start = time.time()
algo_normal_predictor = NormalPredictor()
algo_normal_predictor.fit(trainset)
predictions_normal = algo_normal_predictor.test(testset)
# print(pd.DataFrame(predictions_normal))
end = time.time()
print('Thoi gian thuc hien normal: ', (end - start))
print(accuracy.rmse(predictions_normal))

# BaselineOnly
start = time.time()
algo_Baseline_Only = BaselineOnly()
algo_Baseline_Only.fit(trainset)
predictions_Baseline_Only = algo_Baseline_Only.test(testset)
# print(pd.DataFrame(predictions_Baseline_Only))
end = time.time()
print('Thoi gian thuc hien BaselineOnly: ', (end - start))
print(accuracy.rmse(predictions_Baseline_Only))

# KNNBasic
start = time.time()
algo = KNNBasic()
algo.fit(trainset)
predictions = algo.test(testset)
# print(pd.DataFrame(predictions))
end = time.time()
print('Thoi gian thuc hien KNN_Basic: ', (end - start))
print(accuracy.rmse(predictions))

# KNNWithMeans
start = time.time()
algo = KNNWithMeans()
algo.fit(trainset)
predictions = algo.test(testset)
# print(pd.DataFrame(predictions))
end = time.time()
print('Thoi gian thuc hien KNN_WithMeans: ', (end - start))
print(accuracy.rmse(predictions))

# KNNWithZScore
start = time.time()
algo = KNNWithZScore()
algo.fit(trainset)
predictions = algo.test(testset)
# print(pd.DataFrame(predictions))
end = time.time()
print('Thoi gian thuc hien KNN_WithZScore: ', (end - start))
print(accuracy.rmse(predictions))

# KNNBaseline
start = time.time()
algo = KNNBaseline()
algo.fit(trainset)
predictions = algo.test(testset)
# print(pd.DataFrame(predictions))
end = time.time()
print('Thoi gian thuc hien KNN_Baseline: ', (end - start))
print(accuracy.rmse(predictions))

# NMF
start = time.time()
algo = NMF()
algo.fit(trainset)
predictions = algo.test(testset)
# print(pd.DataFrame(predictions))
end = time.time()
print('Thoi gian thuc hien NMF: ', (end - start))
print(accuracy.rmse(predictions))

#SlopeOne
start = time.time()
algo = SlopeOne()
algo.fit(trainset)
predictions = algo.test(testset)
# print(pd.DataFrame(predictions))
end = time.time()
print('Thoi gian thuc hien SlopeOne: ', (end - start))
print(accuracy.rmse(predictions))

#CoClustering
start = time.time()
algo = CoClustering()
algo.fit(trainset)
predictions = algo.test(testset)
# print(pd.DataFrame(predictions))
end = time.time()
print('Thoi gian thuc hien CoClustering: ', (end - start))
print(accuracy.rmse(predictions))

# SVDpp
start = time.time()
algo = SVDpp()
algo.fit(trainset)
predictions = algo.test(testset)
# print(pd.DataFrame(predictions))
end = time.time()
print('Thoi gian thuc hien SVDpp: ', (end - start))
print(accuracy.rmse(predictions))
