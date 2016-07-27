from Mondrian import *
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def k(x1,x2):
	return np.dot(x1,x2.T)



X = np.genfromtxt('/home/saket/Desktop/train131.csv', delimiter = ',')
Xtest = np.genfromtxt('/home/saket/Desktop/test3.csv', delimiter=',')
train_in_shape = np.shape(X)
y = X[:,[0]]
# X = X[:,1:train_in_shape[0]:1]
X = X[:,[1,2]]
test_in_shape = np.shape(Xtest)
ytest = Xtest[:,[0]]
# Xtest = Xtest[:,1:test_in_shape[0]:1]
Xtest = Xtest[:,[1,2]]

posi = np.where(ytest == 1)
negi = np.where(ytest == -1)
# print(len(posi[0]))
# print(len(negi[0]))
plt.plot(Xtest[posi[0],[0]], Xtest[posi[0],[1]],'ro')
# plt.show()
plt.plot(Xtest[negi[0],[0]], Xtest[negi[0],[1]], 'b+')
plt.show()
# print("shape of X_train: ",train_in_shape)
print("shape of X_test: ",np.shape(Xtest))

cal_train_kernel = np.zeros((train_in_shape[0], train_in_shape[0]), dtype = 'float')
cal_test_kernel = np.zeros((test_in_shape[0], train_in_shape[0]), dtype='float')
# print("shape of cla_train:",np.shape(cal_train_kernel))
# print("shape of cla_test:",np.shape(cal_test_kernel))

for m in range(100):
	print("Iteration:",m)
	test_features, train_features = main(X,Xtest,7)
	cal_train_kernel += k(train_features, train_features) 
	cal_test_kernel += k(test_features, train_features)

cal_test_kernel /= 100
cal_train_kernel /= 100
# print(cal_train_kernel)
# print(np.shape(X))
clf = SVC(C=.05,kernel='linear')
clf.fit(cal_train_kernel,y.ravel())
print(clf.score(cal_train_kernel,y.ravel()))
print(clf.score(cal_test_kernel,ytest.ravel()))
