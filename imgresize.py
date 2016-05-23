from PIL import Image, ImageEnhance
import os
os.chdir('/Users/Thomas/git/thesis/images6')
dim = 3

for i in os.listdir(os.getcwd()):
	if i.endswith('png'):
		if i.startswith('X') or i.startswith('O') or i.startswith('N'):
			img = Image.open(i).resize((dim,dim), Image.ANTIALIAS).convert('LA')
		elif i.startswith('g'):
			img = Image.open(i).resize((dim*3,dim*3), Image.ANTIALIAS).convert('LA')
		bright = ImageEnhance.Brightness(img)
		img = bright.enhance(1.5)
		contrast = ImageEnhance.Contrast(img)
		img = contrast.enhance(3)
		img.save('smalls/'+i)
	else:
		continue