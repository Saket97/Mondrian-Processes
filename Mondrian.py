import numpy as np
import random
import copy
import matplotlib.pyplot as plt
# first column as x axis and second column as y axis
class node():
	def __init__(self):
		self.coordinates = [] #[[x1,x2], [y1,y2]]
		self.trainData = None  # train data without output
		self.testData = None   # test data without output
		self.left = None
		self.right = None
		self.cutPoint = []
		self.dimension = None
		self.dimValue = None
		self.final = None

	def linearDimension(self):
		return abs(self.coordinates[0][0]-self.coordinates[0][1]) +  abs(self.coordinates[1][1]-self.coordinates[1][0])

	def generate_cut(self):
		d = len(self.coordinates)
		m = 0
		for dim in self.coordinates:
			m = m + abs(dim[1] - dim[0])
		cutd = random.uniform(0,m)
		i = 0
		sum1 = abs(self.coordinates[0][1] - self.coordinates[0][0])
		while cutd > sum1:
			i = i + 1
			sum1 = sum1 + abs(self.coordinates[i][1] - self.coordinates[i][0])
			
		
		sum1 -= abs(self.coordinates[i][1]-self.coordinates[i][0])
		rem = cutd-sum1
		self.dimension = i
		self.dimValue = self.coordinates[i][0] + rem
		print("dimension:%d dim-coordinate:%f"%(i,self.dimValue))
		if not i:
			self.cutPoint.append([self.dimValue, self.dimValue])
			self.cutPoint.append(self.coordinates[1])
		else:
			self.cutPoint.append(self.coordinates[0])
			self.cutPoint.append([self.dimValue,self.dimValue])
		plt.plot(self.cutPoint[0], self.cutPoint[1])

def newnode(root):
	# print("saket")
	l = node()
	r = node()
	
	li = np.where((root.trainData)[:,[root.dimension]] <= root.dimValue)
	ri = np.where((root.trainData)[:,[root.dimension]] > root.dimValue)
	l.trainData = (root.trainData)[li[0],:]
	r.trainData = (root.trainData)[ri[0],:]
	
	li1 = np.where((root.testData)[:,[root.dimension]] <= root.dimValue)
	ri1 = np.where((root.testData)[:,[root.dimension]] > root.dimValue)
	l.testData = (root.testData)[li1[0],:]
	r.testData = (root.testData)[ri1[0],:]
	
	l.coordinates = copy.deepcopy(root.coordinates)
	l.coordinates[root.dimension][1] = root.cutPoint[root.dimension][0]
	r.coordinates = copy.deepcopy(root.coordinates)
	r.coordinates[root.dimension][0] = root.coordinates[root.dimension][0]
	root.left = l
	root.right = r

def MP(root,time, leaves):
	cutTime = random.expovariate(root.linearDimension())
	# print("linearDimension: ",root.linearDimension())
	if cutTime > time:
		global count
		count += len(root.testData)
		leaves.append(root)
		# print("time: ",cutTime)
		return
	else:
		# print("time: ",cutTime)
		global cut
		cut += 1
		root.generate_cut()
		newnode(root)
		MP(root.left, time-cutTime,leaves)
		MP(root.right, time-cutTime,leaves)
		return

def feature(leaves, point_index):
	d = len(leaves)
	# print("leaves: ",len(leaves))
	# print("count: ",count)
	p = 0
	boxes = []
	n_points = 0
	n_test = 0
	for i in range(d):
		if len(leaves[i].trainData) or len(leaves[i].testData):
			leaves[i].final = p
			n_points += len(leaves[i].trainData)
			n_test += len(leaves[i].testData)
			p = p+1
			boxes.append(leaves[i])
	vectorTrain = np.zeros((n_points, p),dtype = 'float')
	vectorTest = np.zeros((n_test, p),dtype = 'float')

	for box in boxes:
		for point in box.trainData:
			vectorTrain[point_index[tuple(point)]][box.final] = 1
		for point in box.testData:
			vectorTest[point_index[tuple(point)]][box.final] = 1

	# test_feature = vectorTest
	# train_feature = vectorTrain
	# print("shape of vectortest: ",np.shape(vectorTest))
	# print("shape of vectorTrain: ",np.shape(vectorTrain))
	print("cuts: ",cut)
	# print(vectorTrain)
	return vectorTest, vectorTrain

def main(X, Xtest, time):
	global cut
	global count
	cut = 0
	count = 0
	root = node()
	root.trainData = X
	root.testData = Xtest
	print("shape of xtest in main: ",np.shape(Xtest))
	x1 = min(np.amin(X[:,[0]]), np.amin(Xtest[:,[0]]))-.05
	x2 = max(np.amax(X[:,[0]]), np.amax(Xtest[:,[0]]))+.1
	y1 = min(np.amin(X[:,[1]]), np.amin(Xtest[:,[1]]))-.05
	y2 = max(np.amax(X[:,[1]]), np.amax(Xtest[:,[1]]))+.1
	plt.figure()
	plt.axis([x1,x2,y1,y2])
	print("x1 x2 y1 y2: ",x1,x2,y1,y2)
	root.coordinates.append([x1,x2])
	root.coordinates.append([y1,y2])
	leaves = []
	MP(root,time,leaves)
	point_index = {}
	train_key = list(map(tuple,X))
	test_key = list(map(tuple,Xtest))
	x_shape = np.shape(X)
	for i in range(x_shape[0]):
		point_index[train_key[i]] = i
	Xtest_shape = np.shape(Xtest)
	for i in range(0,Xtest_shape[0]):
		point_index[test_key[i]] = i
	# plt.show()
	plt.close()
	return feature(leaves, point_index)



cut = 0
count = 0;