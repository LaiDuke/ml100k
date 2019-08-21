import heapq
import time
import pandas as pd
from surprise import *
from surprise import Reader, accuracy


class SymmetricAlgo(AlgoBase):

    def __init__(self, sim_options={}, verbose=True, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        return self

    def switch(self, u_stuff, i_stuff):

        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff


class MyOwnAlgorithm(SymmetricAlgo):
    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, verbose=verbose,
                               **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()

        # for i in range(self.sim.shape[0]):
        #     for j in range(self.sim.shape[1]):
        #         self.sim[i][j] = self.sim[i][j]*self.sim[i][j]*self.sim[i][j]
        self.sim = self.sim ** 3

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        sum_sim = sum_ratings = actual_k = 0
        for (sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details


df_train = pd.read_csv(r'training.csv', sep='\t')
df_train.drop(df_train.columns[[0]], axis=1, inplace=True)
reader = Reader(rating_scale=(0.5, 5))
train_ = Dataset.load_from_df(df_train[['userId', 'movieId', 'rating']], reader)
df_test = pd.read_csv(r'testing.csv', sep='\t')
df_test.drop(df_test.columns[[0]], axis=1, inplace=True)
test_ = Dataset.load_from_df(df_test[['userId', 'movieId', 'rating']], reader)
trainset = train_.build_full_trainset()
testset = test_.build_full_trainset().build_testset()

sim_options = {'name': 'cosine', 'user_based': False}
start = time.time()
algo = MyOwnAlgorithm(sim_options=sim_options)
algo.fit(trainset)
predictions = algo.test(testset)

end = time.time()

print(accuracy.rmse(predictions))





