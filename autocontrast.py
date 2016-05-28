from PIL import Image, ImageEnhance, ImageOps
import os
import numpy as np
os.chdir('/Users/Thomas/git/thesis/normalizetests')
dim = 3
for i in os.listdir(os.getcwd()):
	if i.endswith('png') and i.startswith('g'): 
		img = Image.open(i)
		img = ImageOps.autocontrast(img,ignore=range(0,135)+range(230,256))
		img.save('norm'+i)