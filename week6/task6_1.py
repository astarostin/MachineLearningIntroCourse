from skimage.io import imread
from skimage.io import imsave
from skimage import img_as_float
from sklearn.cluster import KMeans
import numpy as np
from math import log10

def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()


def psnr(image1, image2):
	err = np.sum((img_as_float(image1) - img_as_float(image2)) ** 2)
	err /= float(image1.shape[0] * image1.shape[1] * 3)
	return 20 * log10(1) - 10 * log10(err)


image1 = imread('parrots.jpg')
data = img_as_float(image1)
h = data.shape[0]
w = data.shape[1]
data = data.reshape(h * w, 3)

for k in xrange(2, 21):
	km = KMeans(init='k-means++', random_state=241, n_clusters=k)
	clstr = km.fit_predict(data)
	clstr = clstr.reshape(h, w)
	#data = data.reshape(h, w, 3)

	points_by_cluster = {n: [] for n in xrange(km.n_clusters)}

	for i in xrange(h):
		for j in xrange(w):
			points_by_cluster[clstr[i, j]].append(image1[i, j])

	mean_by_cluster = {n: np.mean(points_by_cluster[n], axis=0) for n in xrange(km.n_clusters)}
	median_by_cluster = {n: np.median(points_by_cluster[n], axis=0) for n in xrange(km.n_clusters)}

	image2 = image1.copy()
	for i in xrange(h):
		for j in xrange(w):
			image2[i, j] = mean_by_cluster[clstr[i, j]]
	imsave('parrots_mean_%s.jpg' % k, image2)

	diff = psnr(image1, image2)
	print 'k = %d, psnr = %.4f' % (k, diff)
	if diff > 20:
		out('6_1.txt', str(k))
		break

	# image3 = image1.copy()
	# for i in xrange(h):
	# 	for j in xrange(w):
	# 		image3[i, j] = median_by_cluster[clstr[i, j]]
	# imsave('parrots_median.jpg', image3)