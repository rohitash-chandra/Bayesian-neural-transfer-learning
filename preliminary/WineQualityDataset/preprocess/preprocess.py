import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

def getdata(file):
    data = np.genfromtxt(file, delimiter=';', skip_header=1)
    x = data[:, :-1]
    y = data[:, data.shape[1]-1].astype(int)
    yd = np.identity(10)
    yd = np.vstack([[0 for i in range(10)], yd])
    y = yd[y, :]

    sc_X = StandardScaler()
    x1 = sc_X.fit_transform(x)
    x = normalize(x1, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

    traindata = np.c_[X_train,y_train]
    testdata = np.c_[X_test, y_test]

    return [traindata, testdata]



train,test = getdata('../winequality-red.csv')
np.savetxt('winequality-red-train.csv', train, delimiter = ',')
np.savetxt('winequality-red-test.csv', test, delimiter = ',')
