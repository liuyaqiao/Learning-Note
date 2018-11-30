class union_find:
	def __init__(self, n):
		self.father = [i for i in range(n)]

	def find(self, p):
		if self.father[p] == p:
			return p
		self.father[p] = self.find(self.father[p])
		return self.father[p]

	def union(self, p, q):
		a = self.find(p)
		print(a)
		b = self.find(q)
		print(b)
		if a != b:
			self.father[b] = a 
	
	def connect(self, p, q):
		self.father[q] = p

if __name__ == '__main__':
	s = union_find(5)

	s.connect(1,2)
	# s.connect(3,4)
	s.father[3] = 2
	print(s.find(3))
	print(s.father)