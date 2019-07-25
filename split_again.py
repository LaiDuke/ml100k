from surprise import Dataset
from surprise import Reader
import pandas as pd


from sklearn.model_selection import train_test_split
df = pd.read_csv(r'C:\Users\ADMIN\PycharmProjects\mvl100k\ml-100k\u.data', sep = '\t')

df.columns = ['userId', 'movieId', 'rating', 'timest']
print(df)
reader = Reader(rating_scale=(0.5, 5))
train, test = train_test_split(df, test_size = 0.25)
print(train)
print(test)
train.to_csv('training.csv', sep= '\t')
test.to_csv('testing.csv', sep= '\t')

train_ = Dataset.load_from_df(train[['userId','movieId','rating']],reader)
test_ = Dataset.load_from_df(test[['userId','movieId','rating']],reader)

trainset = train_.build_full_trainset()
testset = test_.build_full_trainset().build_testset()


print(trainset)
print(testset)