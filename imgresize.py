from PIL import Image, ImageEnhance
import os
os.chdir('/Users/Thomas/git/thesis/images7')
dim = 5

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
		if i.startswith('X') or i.startswith('O'):
			img = Image.open(i, 'r')
			red, green, blue, alpha = img.split()
			img = green.resize((dim,dim), Image.ANTIALIAS).convert('LA')
			bright = ImageEnhance.Brightness(img)
			#img = bright.enhance(1.4)
		elif i.startswith('N'):
			img = Image.open(i).resize((dim,dim), Image.ANTIALIAS).convert('LA')
			bright = ImageEnhance.Brightness(img)
			#img = bright.enhance(1.7)
		#elif i.startswith('g'):
		#	img = Image.open(i).resize((dim*3,dim*3), Image.ANTIALIAS).convert('LA')
		contrast = ImageEnhance.Contrast(img)
		#img = contrast.enhance(4.5)

		img.save('smalls/'+i)

		#img = Image.open('smalls/'+i).convert('RGBA')
		#img = np.array(img)
		#img = Image.fromarray(normalize(img).astype('uint8'),'RGBA')
		#img.save('smalls/'+i)
	else:
		continue