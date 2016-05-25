from PIL import Image, ImageEnhance
import os
os.chdir('/Users/Thomas/git/thesis/images6')
dim = 3

def normalize(arr):
    arr = arr.astype('float')
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

for i in os.listdir(os.getcwd()):
	if i.endswith('png'):
		if i.startswith('X') or i.startswith('O') or i.startswith('N'):
			img = Image.open(i).resize((dim,dim), Image.ANTIALIAS).convert('LA')
			bright = ImageEnhance.Brightness(img)
			img = bright.enhance(1.3)
		#elif i.startswith('N'):
		#	img = Image.open(i).resize((dim,dim), Image.ANTIALIAS).convert('LA')
		#	bright = ImageEnhance.Brightness(img)
		#	img = bright.enhance(1.55)
		elif i.startswith('g'):
			img = Image.open(i).resize((dim*3,dim*3), Image.ANTIALIAS).convert('LA')
		contrast = ImageEnhance.Contrast(img)
		img = contrast.enhance(3)

		img.save('smalls/'+i)

		#img = Image.open('smalls/'+i).convert('RGBA')
		#img = np.array(img)
		#img = Image.fromarray(normalize(img).astype('uint8'),'RGBA')
		#img.save('smalls/'+i)
	else:
		continue