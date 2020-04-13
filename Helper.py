import os

'''
	Utility functions that are needed in multiple scripts
'''

# Creates a folder at path location
# does not recursively create path
def make_folder(path):
	try:
		os.mkdir(path)
	except FileNotFoundError:
		print("{} path can't be created".format(path))
	except FileExistsError:
		print("{} folder already exists".format(path))

