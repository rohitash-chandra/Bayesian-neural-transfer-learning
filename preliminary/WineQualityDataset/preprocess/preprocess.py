import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

def getdata(file):
    data = np.genfromtxt(file, delimiter=';', skip_header=1)
    x = data[:, :-1]
    y = data[:, data.shape[1]-1].astype(int)
    #print data
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


def testdata(file):
    data = np.genfromtxt(file, delimiter=',')
    data = data[:, -10:]
    y = np.zeros((data.shape[0],))
    yd = np.vstack([[0 for i in range(10)],np.identity(10)])
    #print yd
    j = 0
    for row in data:
        for index in range(yd.shape[0]):
            if np.array_equal(row, yd[index]):
                y[j] = index
                j += 1
                print str(row)+ " " + str(index)
            #else: print 'no match' + str(row) 


wine = ['winequality-white', 'winequality-red']
for data in wine:
    train,test = getdata('../'+data+'.csv')
    np.savetxt(data+'-train.csv', train, delimiter = ',')
    np.savetxt(data+'-test.csv', test, delimiter = ',')
    
    print "Testing Train data:\n"
    testdata(data+'-train.csv')
    
    print "Testing Test data:\n"
    testdata(data+'-test.csv')
