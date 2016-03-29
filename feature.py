import random
import copy
random.seed(2)
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
	for i in range(d):
		s = s + abs(root.coordinates[i][1] - root.coordinates[i][0])
	return s
				
#returns whch dimension has maximum length of the box	
def max_dimension(root):
	d = len(root.coordinates)
	m = 0
	for i in range(d):
		if abs(root.coordinates[i][1] - root.coordinates[i][0]) > abs(root.coordinates[m][1] - root.coordinates[m][0]):
			m = i
		else:
			pass
	return m

def find_cut(root):
	global number
	number = number+1
	d = max_dimension(root)
	cut = random.uniform(root.coordinates[d][0], root.coordinates[d][1])
	coordi = [d,cut]
	return coordi

#t0 is the lifetime of that box		
def mondrian(root , t0):
	t = random.expovariate(linear_dimension(root))
	if t >= t0:
		global boxes
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
	mondrian(root.left , t0 - t)
	mondrian(root.right , t0 - t)
		
def feature(boxes):
	n = len(boxes)
	p = 0
	
	for node in boxes:
		tmp = []
		for i in range(n):
			if i == p:
				tmp.append(1)
			else:
				tmp.append(0)
		for point in node.point_list:
			print("point:",point,"  feature vector:",tmp)
				
		
		p = p+1
			
					
		
root = Node()
root.coordinates = []
root.point_list = []
root.cut_location = []
root.left = []
root.right = []
		
boxes = [] #contains the leaf nodes i.e final boxes after partition

n = int(input("enter the number of points\n"))
d = int(input("enter the dimension of box\n"))
for i in range(d):
	tmp = []
	tmp = list(map(float,input("enter the coordinates of box along %d dimension in the form [a,b]\n"%i).split()))
	root.coordinates.append(tmp)
for i in range(n):
	tmp = []
	tmp = list(map(float,input("enter the coordinates of point\n").split()))
	root.point_list.append(tmp)
t = float(input("enter the time for which you want the process to run\n"))
print(root.coordinates)
print(root.point_list)
"""
n = 2
d = 2
root.coordinates = [[0,5],[0,5]]
root.point_list.append([1,1])
root.point_list.append([2,2])
t = 2.0
"""

number = 0
mondrian(root,t)
feature(boxes)
print(number)

	
