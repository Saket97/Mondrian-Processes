import random
import copy
import time
import numpy as np
import csv
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import multiprocessing as mp
start = time.time()
class Node:
	def _init_(self):
		self.cut_location = [] #first entry contains the cut dimension and scond its value in that dimension
		self.point_list = []   #self-explanatory 
		self.coordinates = []  #its every entry is of the form [a,b] showing its range in that dimension
		self.left = None       #box having points with coordinates having value less than cut
		self.right = None      #box having points with coordinates having value more than cut

def linear_dimension(root):
	d = len(root.coordinates)
	s = 0
	#print('d',d)
	for i in range(d):
		#print('i',i)
		s = s + abs(root.coordinates[i][1] - root.coordinates[i][0])
	return s
				
#returns whch dimension has maximum length of the box	
def max_dimension(root):
	d = len(root.coordinates)
	m = 0
	for dim in root.coordinates:
		m = m + abs(dim[1] - dim[0])
	cutd = random.uniform(0,m)
	i = 0
	sum1 = abs(root.coordinates[0][1] - root.coordinates[0][0])
	while cutd > sum1:
		i = i + 1
		sum1 = sum1 + abs(root.coordinates[i][1] - root.coordinates[i]
		[0])
	
	return i

def find_cut(root):
#	global number
#	number = number+1
	d = max_dimension(root)
	cut = random.uniform(root.coordinates[d][0], root.coordinates[d][1])
	coordi = [d,cut]
	return coordi

#t0 is the lifetime of that box		
def mondrian(root , t0,boxes):
	t = random.expovariate(linear_dimension(root))
	if t >= t0:
		
		boxes.append(root)
		return
	n = len(root.point_list)
	d = len(root.coordinates)
	l = Node()
	r = Node()
	
	l.cut_location = []
	l.point_list = []
	l.coordinates = []
	l.left = None
	l.right = None
	
	r.cut_location = []
	r.point_list = []
	r.coordinates = []
	r.left = None
	r.right = None
	
	root.cut_location = find_cut(root)
	for point in root.point_list:
		if point[root.cut_location[0]] < root.cut_location[1]:
			l.point_list.append(point)
		else:
			r.point_list.append(point)
	l.coordinates = copy.deepcopy(root.coordinates)
	r.coordinates = copy.deepcopy(root.coordinates)
#	print(root.coordinates)
#	print("id of root",id(root))
#	print("id od l",id(l))
	l.coordinates[root.cut_location[0]][1] = root.cut_location[1]
#	print(root.coordinates)
#	print(l.coordinates)
	r.coordinates[root.cut_location[0]][0] = root.cut_location[1]
#	print("l.coordinates:",l.coordinates)
#	print("r.coordinates:",r.coordinates)
#	print("root.coordinates:",root.coordinates)
	root.left = l
	root.right = r
	mondrian(root.left , t0 - t,boxes)
	mondrian(root.right , t0 - t,boxes)
		
def feature(boxes, n_points, point_index):
	n = len(boxes)
	p = 0
	
	for node in boxes:
		if len(node.point_list) != 0:
			p = p+1
	i = 0      #index of box i.e. dimension having 1
	s = 0
	vector = np.zeros((n_points, p),dtype = 'float')
	#print('no. of partitions',len(boxes))
	#print(vector)
	#print(n_points, p )
	for node in boxes:
		if len(node.point_list) != 0:
			for points in node.point_list:
				vector[s][i] = 1
				point_index[tuple(points)] = s
				s = s+1
			i = i+1
		else:
			pass
				
	return vector				
# x1 and x2 are tuples
def k(x1,x2):

	n1 = np.shape(x1)
	n2 = np.shape(x2)
	#print('x1,x2',x1,x2)
	to_return = np.empty((n1[0],n2[0]),dtype = 'float')
	for i in range(n1[0]):
		for j in range(n2[0]):
			to_return[i][j] = np.dot(x1[i],x2[j])
	return to_return

def output(point_index,n,true_output,true_input):
	y = np.empty(n,dtype='int16')
	for i in range(n):
		try:
			d = tuple(true_input[i])
			y[point_index[d]] = true_output[i]
		except:
#			print('n,i,d',n,i,d)
#			print('true_input,true_output,point_index',len(true_input),len(true_output),len(point_index))
			pass
	return y	

def read(input_file,x,y):
	data = csv.reader(input_file,skipinitialspace = True,quoting = csv.QUOTE_NONNUMERIC,)
	for lin in data:
		tmp = lin[1:]
		x.append(tmp)
		y.append(lin[0])
		
	del data
			
def separate(vector,y1,x_t,point_index,test_feature,test_output):
	i = 0
	tmp = []
#	print('y1',len(y1))
	for element in x_t:
		n = point_index[tuple(element)]
		try:
			test_feature[i] = vector[n]
			test_output[i] = y1[n]
		except:
#			print('test_feature,test_output,vector,y1,i,n',len(test_feature),len(test_output),len(vector),len(y1),i,n)
			pass
		tmp.append(n)
		i = i+1
#	print('id',id(vector))
	vector = np.delete(vector,tmp,axis = 0)
	y1 = np.delete(y1,tmp)
#	print('id',id(vector))
#	print('tmp,test_feature,test_output',tmp,test_feature,test_output)
#	print('vector,y1',vector,y1)
	
	clf = SVC(kernel = k)
	clf.fit(vector,y1)
	
	c = np.array(clf.predict(test_feature))
#	print('test_feature',test_feature)
#	print('predict',c)
#	print('accuracy is %f%%' % (clf.score(test_feature,test_output)*100))
	return c
	
def main(x,y,x_t,y_t,coordi):				
	root = Node()
	root.coordinates = []
	root.point_list = []
	root.cut_location = []	
	root.left = []
	root.right = []
		
	boxes = [] #contains the leaf nodes i.e final boxes after partition

	
	root.point_list = x + x_t
	n = len(root.point_list)
	n_d = np.shape(root.point_list)
	d = n_d[1]
	"""
	for i in range(d):
		tmp = []
		tmp = list(map(float,input("enter the coordinates of box along %d dimension in the form [a,b]\n"%i).split()))
		root.coordinates.append(tmp)
	print('root.coordinates',root.coordinates)
	print('root.point_list',root.point_list)
	"""
	root.coordinates = coordi

	t = 1
	
	
	point_index = {}
#	number = 0
	mondrian(root,t,boxes)
	vector = feature(boxes,n, point_index)
#	print('vector',vector)
#	print('number of cuts',number)
	
	
	true_output = y + y_t
	y1 = output(point_index,n,true_output,root.point_list) #true_input is root.point_list true output and input should be read from csv file
	
	#print('y',y1)
#	print('point_index',len(point_index))
	n_vector = np.shape(vector)
	test_feature = np.empty((len(x_t),n_vector[1]),dtype = 'int16')
	test_output = np.empty((len(x_t)),dtype = 'int16')
	
	return separate(vector,y1,x_t,point_index,test_feature,test_output)
	
if __name__ == '__main__':
	input_file = open('/home/saket/Desktop/train131.csv','r',newline='')
	input_test = open('/home/saket/Desktop/test3.csv','r',newline = '')
	
	x_t = []
	y_t = []
	x = []
	y = []
	read(input_test,x_t,y_t)
	read(input_file,x,y)
	input_file.close()
	input_test.close()	
	del input_test
	del input_file
#	print(main(x[:],y[:],x_t[:],y_t[:]))
	pred = []
	coordi = [[0,0.5],[0,0.5],[0,0.5],[0,0.5],[0,0.5],[0,0.5]]
	for i in range(100):
		pred.append(main(x[:],y[:],x_t[:],y_t[:],coordi))
	predic = np.array(pred)
	a = np.sum(predic,axis=0)
	a = a/100
	i = np.shape(a)
	t_output = np.array(y_t)
	for j in range(i[0]):
		if a[j] >= 0:
			a[j] = 1
		else:
			a[j] = -1
	print(a)
	print('accuracy is' ,accuracy_score(a,t_output))
	
	
	
