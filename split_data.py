from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
import pandas as pd


data = Dataset.load_builtin('ml-100k')

trainset100k, testset100k = train_test_split(data, test_size=.25)

print('Number of test data is:', len(testset100k))

from sklearn.model_selection import train_test_split
df = pd.read_csv(r'C:\Users\ADMIN\PycharmProjects\mvl100k\ml-20m\ml-20m\ratings.csv')
reader = Reader(rating_scale=(0.5, 5))
train, test = train_test_split(df, test_size = 0.25)

train_ = Dataset.load_from_df(train[['userId','movieId','rating']],reader)
test_ = Dataset.load_from_df(test[['userId','movieId','rating']],reader)

trainset = train_.build_full_trainset()
testset = test_.build_full_trainset().build_testset()



