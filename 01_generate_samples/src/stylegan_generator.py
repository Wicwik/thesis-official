# dnnlib is custom nvidia lib for StyleGAN2 project
# its included in the SyleGAN2 repo submodule
import dnnlib.tflib as tflib
import dnnlib

import pickle
import numpy as np

# simple generator of images
# partialy adapted from official implemetantion generate images file https://github.com/NVlabs/stylegan2/blob/master/run_generator.py
class StyleGANGenerator:
	def __init__(self, path_or_url):
		self.path_or_url = path_or_url
		self.G, self.D, self.Gs = self._get_sylegan_networks(path_or_url)

	# method that generated images from z
	# z has size (m, 512), where m is number of input vectors
	def get_images(self, z, truncation_psi = None):
		noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]

		Gs_kwargs = dnnlib.EasyDict()
		Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
		Gs_kwargs.randomize_noise = False
		if truncation_psi is not None:
			Gs_kwargs.truncation_psi = truncation_psi
			
		rnd = np.random.RandomState()
		tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
		imgs = self.Gs.run(z, None, **Gs_kwargs)

		return imgs

	# init tensorflow and get pretrained networks from url
	def _get_sylegan_networks(self, path_or_url):
	    if dnnlib.util.is_url(path_or_url):
	        stream = dnnlib.util.open_url(path_or_url, cache_dir='.stylegan2-cache')
	    else:
	        stream = open(path_or_url, 'rb')

	    tflib.init_tf()
	    with stream:
	        G, D, Gs = pickle.load(stream, encoding='latin1')

	    return G, D, Gs