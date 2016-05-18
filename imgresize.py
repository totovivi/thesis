from PIL import Image, ImageEnhance
import os
os.chdir('/Users/Thomas/git/thesis/images5')
dim = 5

for i in os.listdir(os.getcwd()):
	if i.startswith('X') or i.startswith('O') or i.startswith('N'):
		img = Image.open(i).resize((dim,dim), Image.ANTIALIAS)
		bright = ImageEnhance.Brightness(img)
		img = bright.enhance(1.7)
		contrast = ImageEnhance.Contrast(img)
		img = contrast.enhance(10)
		img.save('smalls/'+i)
	elif i.startswith('g'):
		img = Image.open(i).resize((dim*3,dim*3), Image.ANTIALIAS)
		bright = ImageEnhance.Brightness(img)
		img = bright.enhance(1.5)
		contrast = ImageEnhance.Contrast(img)
		img = contrast.enhance(40)
		img.save('smalls/'+i)
	else:
		continue