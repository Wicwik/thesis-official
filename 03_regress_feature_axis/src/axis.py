import numpy as np
import sklearn.linear_model as linear_model

# helper functions to regress feature axis
# partialy adapted from https://github.com/SummitKwan/transparent_latent_gan/blob/master/src/tl_gan/feature_axis.py


def find_feature_axis(z, y, method='linear', **kwargs_model):
	if method == 'linear':
		# train linear regression without transforming labels
		model = linear_model.LinearRegression(**kwargs_model)
		model.fit(z, y)
	elif method == 'tanh':
		def arctanh_clip(y):
			return np.arctanh(np.clip(y, np.tanh(0), np.tanh(3)))

		# train linear regression with labels transformed to arctanh
		model = linear_model.LinearRegression(**kwargs_model)
		model.fit(z, arctanh_clip(y))

	return model.coef_.transpose() # return transposed coeficients

# function that normalizes the vector
def normalize_feature_axis(feature_slope):
	return feature_slope / np.linalg.norm(feature_slope, ord=2, axis=0, keepdims=True)