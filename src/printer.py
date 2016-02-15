import sys
import time

class Printer(object):
	def __init__(self, wait_time):
		self.wait_time = wait_time
		self.t = time.time()
		self.last_length = 0
		
	def overwrite(self, message='', wait=True):
		if time.time() - self.t >= self.wait_time or not wait:
			sys.stdout.write(' '*self.last_length + '\r')
			sys.stdout.flush()
			sys.stdout.write(message)
			sys.stdout.flush()
			self.t = time.time()
			self.last_length = len(message)

	def clear(self):
		self.overwrite(wait=False)