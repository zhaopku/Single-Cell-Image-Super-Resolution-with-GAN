import torch
from tqdm import tqdm
import argparse
import os
import torch.optim as optimizer
from models.data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, TestDatasetFromFolder, display_transform
from torch.utils.data import DataLoader
from models.model_srcnn import ModelSRCNN
from models import utils
from models.model_gen import SRGANGenerator
import pytorch_ssim
import math
import torchvision.utils

class Train:
	def __init__(self):
		self.args = None
		self.training_set = None
		self.val_set = None
		self.train_loader = None
		self.val_loader = None
		self.model = None

		self.result_dir = None
		self.out_image_dir = None

		self.naive_results = None
		self.naive_results_computed = False

	@staticmethod
	def parse_args(args):
		parser = argparse.ArgumentParser()

		data_args = parser.add_argument_group('Dataset options')
		data_args.add_argument('--data_dir', default='./data/ds_dapi_mini')
		data_args.add_argument('--train_dir', default='train')
		data_args.add_argument('--val_dir', default='val')
		data_args.add_argument('--test_dir', default='test')

		data_args.add_argument('--result_dir', default='./result')

		# neural network options
		nn_args = parser.add_argument_group('Network options')
		nn_args.add_argument('--crop_size', default=30, type=int, help='training images crop size')
		nn_args.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 6, 8],
		                    help='super resolution upscale factor')
		nn_args.add_argument('--model', type=str, default='SRGAN_GEN', help='string to specify model')
		nn_args.add_argument('--in_channels', type=int, default=1)

		# training options
		training_args = parser.add_argument_group('Training options')
		training_args.add_argument('--batch_size', type=int, default=10)
		training_args.add_argument('--n_save', type=int, default=5, help='number of test images to save on disk')
		training_args.add_argument('--epochs', type=int, default=200, help='number of training epochs')
		training_args.add_argument('--lr', type=float, default=0.001, help='learning rate')

		return parser.parse_args(args)

	def construct_data(self):
		self.training_set = TrainDatasetFromFolder(os.path.join(self.args.data_dir, self.args.train_dir),
		                                           crop_size=self.args.crop_size,
		                                   upscale_factor=self.args.upscale_factor)

		self.val_set = ValDatasetFromFolder(os.path.join(self.args.data_dir, self.args.val_dir),
		                                   upscale_factor=self.args.upscale_factor)

		self.train_loader = DataLoader(dataset=self.training_set, num_workers=0, batch_size=self.args.batch_size, shuffle=True)
		self.val_loader = DataLoader(dataset=self.val_set, num_workers=0, batch_size=self.args.batch_size, shuffle=False)

	def construct_model(self):
		if self.args.model == 'SRCNN':
			self.model = ModelSRCNN(args=self.args)
		elif self.args.model == 'SRGAN_GEN':
			self.model = SRGANGenerator(args=self.args)

		elif self.args.model == 'SRGAN':
			raise NotImplementedError
		else:
			print('Invalid model string: {}'.format(self.args.model))

		self.optimizer = optimizer.Adam(self.model.parameters(), lr=self.args.lr)

		# average over all the pixels in the batch
		self.loss = torch.nn.MSELoss(reduction='elementwise_mean')

		print('{}, #param = {}'.format(self.args.model, sum(param.numel() for param in self.model.parameters())))

	def construct_out_dir(self):
		self.result_dir = utils.construct_dir(prefix=self.args.result_dir, args=self.args)
		self.out_image_dir = os.path.join(self.result_dir, 'images')
		self.model_dir = os.path.join(self.result_dir, 'models')
		self.out_path = os.path.join(self.result_dir, 'result.txt')

		if not os.path.exists(self.out_image_dir):
			os.makedirs(self.out_image_dir)

		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)

	def main(self, args=None):
		print('PyTorch Version {}, GPU enabled {}'.format(torch.__version__, torch.cuda.is_available()))
		self.args = self.parse_args(args=args)

		self.construct_data()

		self.construct_model()

		self.construct_out_dir()


		with open(self.out_path, 'w') as self.out:
			self.train_loop()

	def train_loop(self):

		# put model to GPU if available
		if torch.cuda.is_available():
			self.model.cuda()
			self.loss = self.loss.cuda()

		for e in range(self.args.epochs):
			# switch the model to training mode
			self.model.train()
			train_loss = 0
			for idx, (lr_image, hr_image) in enumerate(tqdm(self.train_loader)):
				# put data to GPU
				if torch.cuda.is_available():
					lr_image = lr_image.cuda()
					hr_image = hr_image.cuda()

				# sr_image, should have the same shape as hr_image
				sr_image = self.model(lr_image)

				loss = self.loss(input=sr_image, target=hr_image)

				train_loss += loss

				self.model.zero_grad()
				loss.backward()
				self.optimizer.step()

			print('Epoch = {}, Train loss = {}'.format(e, train_loss))
			self.out.write('Epoch = {}, Train loss = {}\n'.format(e, train_loss))
			self.out.flush()
			self.validate(epoch=e)

	def validate(self, epoch):
		self.model.eval()
		val_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'val_size': 0}

		if not self.naive_results_computed:
			self.naive_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'val_size': 0}

		val_images = []
		for idx, (lr_image, naive_hr_image, hr_image) in enumerate(tqdm(self.val_loader)):
			# put data to GPU

			cur_batch_size = lr_image.size(0)
			val_results['val_size'] += cur_batch_size

			if torch.cuda.is_available():
				lr_image = lr_image.cuda()
				naive_hr_image = naive_hr_image.cuda()
				hr_image = hr_image.cuda()

			sr_image = self.model(lr_image)

			batch_mse = ((sr_image - hr_image) ** 2).data.mean()
			val_results['mse'] += batch_mse * cur_batch_size
			batch_ssim = pytorch_ssim.ssim(sr_image, hr_image).item()
			val_results['ssims'] += batch_ssim * cur_batch_size
			val_results['psnr'] = 10 * math.log10(1 / (val_results['mse'] / val_results['val_size']))
			val_results['ssim'] = val_results['ssims'] / val_results['val_size']

			if not self.naive_results_computed:
				naive_batch_mse = ((naive_hr_image - hr_image) ** 2).data.mean()
				self.naive_results['mse'] += naive_batch_mse * cur_batch_size
				naive_batch_ssim = pytorch_ssim.ssim(naive_hr_image, hr_image).item()
				self.naive_results['ssims'] += naive_batch_ssim * cur_batch_size
				self.naive_results['psnr'] = 10 * math.log10(1 / (self.naive_results['mse'] / val_results['val_size']))
				self.naive_results['ssim'] = self.naive_results['ssims'] / val_results['val_size']

			# only save certain number of images

			# transform does not support batch processing
			if idx < self.args.n_save:
				for image_idx in range(cur_batch_size):
					val_images.extend(
						[display_transform()(lr_image[image_idx].data.cpu()),
						 display_transform()(naive_hr_image[image_idx].data.cpu()),
						 display_transform()(hr_image[image_idx].data.cpu()),
						 display_transform()(sr_image[image_idx].data.cpu())])

		# write to out file
		result_line = '\tVal\t'
		for k, v in val_results.items():
			result_line += '{} = {} '.format(k, v)

		if not self.naive_results_computed:
			result_line += '\n'
			for k, v in self.naive_results.items():
				result_line += 'naive_{} = {} '.format(k, v)
			self.naive_results_computed = True

		print(result_line)
		self.out.write(result_line+'\n')
		self.out.flush()
		# save model
		torch.save(self.model.state_dict(), os.path.join(self.model_dir, str(epoch)+'.pth'))

		val_images = torch.stack(val_images)

		# number of out images, 15 is number of sub-images in an output image
		n_out_images = (val_images.size(0) // (4*5) + 1)

		cur_out_image_dir = os.path.join(self.out_image_dir, 'epoch_%d' % epoch)

		if not os.path.exists(cur_out_image_dir):
			os.makedirs(cur_out_image_dir)

		for idx in tqdm(range(n_out_images), desc='saving validating image'):
			image = torch.Tensor(val_images[idx*(4*5):(idx+1)*(4*5)])
			if image.size()[0] < 1:
				break
			image = torchvision.utils.make_grid(image, nrow=4, padding=5)
			save_path = os.path.join(cur_out_image_dir, 'index_%d.jpg' % idx)
			torchvision.utils.save_image(image, save_path, padding=5)


