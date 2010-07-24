''' unit test framework for the FireFly population 
	and FireFly classes
'''

import unittest as ut
import math as m
from FireFly import *

class TestFireFly(ut.TestCase):
	
	def setUp(self):
		''' setup our test suite for the FireFly
			tests
		'''

		# we'll do three self.dimensions
		self.dim = 3

		# create our test function with its bounds
		# f*(x) = 0.0
		of = lambda x: sum([c**2.0 for c in x])
		self.func = ObjFunc(of, [-5.0]*self.dim, [5.0]*self.dim)

		# create our dummy population with alpha = 0.0
		# the reason for alpha = 0.0 is to cull all randomness
		# this will allow for testable, repeatable behavior
		# in actuality, some randomness will alter the 
		# behavior of the population
		self.pop = Population(1, 2, 0.0, 1.0, 1.0)

		# create our fireflies
		self.fly = FireFly(self.func, self.pop, self.dim, [0.0]*self.dim)
		self.fly.val = of([0.0]*self.dim)

		self.ofly = FireFly(self.func, self.pop, self.dim, [1.0]*self.dim)
		self.ofly.val = of([1.0]*self.dim)
	
	def test_CalculateDistance(self):
		''' this tests calculating euclidean distance
		'''

		# sqrt( (0 - 1)**2.0 + (0 - 1)**2.0 + (0 - 1)**2.0 ) == sqrt(3.0)
		cval = m.sqrt(3.0)

		tval = self.fly.calculate_dist(self.ofly)

		self.assertAlmostEqual(cval, tval, msg='Testing the distance')

		# these should be the same no matter which order
		tval = self.ofly.calculate_dist(self.fly)

		self.assertAlmostEqual(cval, tval, msg='Testing the distance')
		
	def test_CopyFireFly(self):
		''' this test the copy function of the firefly
		'''

		f = FireFly(self.func, None, self.dim, [5.0]*self.dim)

		f.copy(self.fly)

		self.assertEqual(f.val, self.fly.val, msg='Testing Copy Function Values')

		for x, y in zip(f.coords, self.fly.coords):
			self.assertAlmostEqual(x, y, msg='Testing coordinates')

		f.copy(self.ofly)

		self.assertAlmostEqual(f.val, self.ofly.val, msg='Testing Copy Function Values')

		for x, y in zip(f.coords, self.ofly.coords):
			self.assertAlmostEqual(x, y, msg='Testing coordinates')

	def test_Eval(self):
		''' this tests the evaluation of the firefly
		'''

		cval = self.func.eval([1.0]*self.dim)
		# we have to grab it from the fly
		self.ofly.eval()
		tval = self.ofly.val

		self.assertAlmostEqual(cval, tval, msg='Testing evaluation values')
	

if __name__=='__main__':
	ut.main()
