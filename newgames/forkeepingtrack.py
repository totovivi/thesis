import os

path = '/Users/Thomas/git/thesis/newgames'
files = sorted(os.listdir(path), key=lambda x: os.path.getctime(os.path.join(path, x)))
list = [input(files)]
print list