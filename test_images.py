import torch, torchvision
import numpy as np
import argparse
from models.data_utils import ValDatasetFromFolder, TestOutDatasetFromFolder, display_transform, create_new_lr_image
from torch.utils.data import DataLoader
from models.model_gen import SRGANGenerator
from models.model_discriminator import Discriminator
import os
from tqdm import tqdm
import pytorch_ssim
import math

class Test:
	def __init__(self):
		self.model_path = None
		# list
		self.test_images = None
		self.in_path = None
		self.out_path = None
		self.model_path = None

		self.out_file = None
		self.out = None

	@staticmethod
	def parse_args(args):
		parser = argparse.ArgumentParser()

		data_args = parser.add_argument_group('Dataset options')
		data_args.add_argument('--in_path', default='./test_in_images')
		data_args.add_argument('--out_path', default='./test_out_images')

		# neural network options
		nn_args = parser.add_argument_group('Network options')
		nn_args.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
		                    help='super resolution upscale factor')
		nn_args.add_argument('--model_dir', type=str, default='saved_models', help='saved model dir')
		nn_args.add_argument('--in_channels', type=int, default=1)

		# training options
		training_args = parser.add_argument_group('Training options')
		training_args.add_argument('--batch_size', type=int, default=10)
		training_args.add_argument('--n_save', type=int, default=10, help='number of batches to save')

		return parser.parse_args(args)

	def construct_dir(self):
		# self.in_path = os.path.join(self.args.in_path, 'up_%d' % self.args.upscale_factor)
		self.in_path = self.args.in_path
		self.out_path = os.path.join(self.args.out_path, 'up_%d' % self.args.upscale_factor)
		self.model_path = os.path.join(self.args.model_dir, 'up_%d.pth' % self.args.upscale_factor)
		self.out_file = os.path.join(self.out_path, 'result.txt')

		if not os.path.exists(self.out_path):
			os.makedirs(self.out_path)

	def construct_data(self):
		self.test_set = ValDatasetFromFolder(self.in_path,
		                                   upscale_factor=self.args.upscale_factor)

		self.test_loader = DataLoader(dataset=self.test_set, num_workers=1, batch_size=self.args.batch_size, shuffle=False)

	def main(self, args=None):
		self.args = self.parse_args(args=args)
		self.generator = SRGANGenerator(args=self.args)
		self.discriminator = Discriminator(args=self.args)
		self.construct_dir()
		if torch.cuda.is_available():
			(generate_state_dict, discriminator_state_dict) = torch.load(self.model_path, map_location='gpu')
		else:
			(generate_state_dict, discriminator_state_dict) = torch.load(self.model_path, map_location='cpu')

		self.generator.load_state_dict(generate_state_dict)
		self.discriminator.load_state_dict(discriminator_state_dict)

		self.construct_data()
		self.mse_loss = torch.nn.MSELoss(reduction='elementwise_mean')


		with open(self.out_file, 'w') as self.out:
			self.test_loop()

	def test_loop(self):
		# put to GPU
		if torch.cuda.is_available():
			self.generator.cuda()

		# mse, ssim, and psnr are not available at current settings
		test_results = {'mse_loss': 0, 'D_G_z':0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'n_samples': 0, 'D_x': 0}

		naive_results_computed = False
		naive_results = {'mse_loss': 0, 'D_G_z': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'n_samples': 0}

		with torch.no_grad():
			self.generator.eval()
			self.discriminator.eval()
			test_images = []
			for idx, (lr_image, naive_hr_image, hr_image) in enumerate(tqdm(self.test_loader)):
				if idx >= self.args.n_save:
					break
				cur_batch_size = lr_image.size(0)
				test_results['n_samples'] += cur_batch_size

				if torch.cuda.is_available():
					lr_image = lr_image.cuda()
				hr_probs, log_hr_probs = self.discriminator(hr_image)

				test_results['D_x'] += hr_probs.data.cpu().sum()

				sr_image = self.generator(lr_image)
				sr_probs, log_sr_probs = self.discriminator(sr_image)
				test_results['D_G_z'] += sr_probs.data.cpu().sum()

				lr_image = create_new_lr_image(lr_image, hr_image)
				sr_image = create_new_lr_image(sr_image, hr_image)
				naive_hr_image = create_new_lr_image(naive_hr_image, hr_image)
				naive_sr_probs, naive_log_sr_probs = self.discriminator(naive_hr_image)
				naive_results['D_G_z'] += naive_sr_probs.data.cpu().sum()

				mse_loss = self.mse_loss(input=sr_image, target=hr_image)
				test_results['mse_loss'] += mse_loss.data.cpu() * cur_batch_size

				batch_ssim = pytorch_ssim.ssim(sr_image, hr_image).item()
				test_results['ssims'] += batch_ssim * cur_batch_size
				test_results['psnr'] = 10 * math.log10(1 / (test_results['mse_loss'] / test_results['n_samples']))
				test_results['ssim'] = test_results['ssims'] / test_results['n_samples']

				naive_mse_loss = self.mse_loss(input=naive_hr_image, target=hr_image).data.cpu()
				naive_results['mse_loss'] += naive_mse_loss * cur_batch_size
				naive_batch_ssim = pytorch_ssim.ssim(naive_hr_image, hr_image).item()
				naive_results['ssims'] += naive_batch_ssim * cur_batch_size
				naive_results['psnr'] = 10 * math.log10(1 / (naive_results['mse_loss'] / test_results['n_samples']))
				naive_results['ssim'] = naive_results['ssims'] / test_results['n_samples']

				for image_idx in range(cur_batch_size):
					test_images.extend(
						[display_transform()(lr_image[image_idx].data.cpu()),
						 display_transform()(naive_hr_image[image_idx].data.cpu()),
						 display_transform()(hr_image[image_idx].data.cpu()),
						 display_transform()(sr_image[image_idx].data.cpu())])

			test_results['D_G_z'] /= test_results['n_samples']
			test_results['D_x'] /= test_results['n_samples']
			naive_results['D_G_z'] /= test_results['n_samples']

			test_results['mse_loss'] /= test_results['n_samples']
			naive_results['mse_loss'] /= test_results['n_samples']

			# write to out file
			result_line = '\tTest\n'
			for k, v in test_results.items():
				result_line += '{} = {}, '.format(k, v)

			result_line += '\n'
			for k, v in naive_results.items():
				result_line += 'naive_{} = {} '.format(k, v)
			print(result_line)
			self.out.write(result_line+'\n')
			self.save_image(test_images)

	def save_image(self, images):
		images = torch.stack(images)

		# number of out images, 2*5 is number of sub-images in an output image
		n_out_images = (images.size(0) // (4*5) + 1)

		for idx in tqdm(range(n_out_images), desc='saving validating image'):
			image = torch.Tensor(images[idx*(4*5):(idx+1)*(4*5)])
			if image.size()[0] < 1:
				break
			image = torchvision.utils.make_grid(image, nrow=4, padding=5)
			save_path = os.path.join(self.out_path, 'index_%d.jpg' % idx)
			torchvision.utils.save_image(image, save_path, padding=5)


if __name__ == '__main__':
	test = Test()
	test.main()