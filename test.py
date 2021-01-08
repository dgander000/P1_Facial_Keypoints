import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from data_load import Rescale, RandomCrop, Normalize, ToTensor
from data_load import FacialKeypointsDataset
from models import Net


def get_train_test_data():
	data_transform = transforms.Compose([Rescale(250),
									 RandomCrop(224),
									 Normalize(),
									 ToTensor()])

	transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
												root_dir='data/training/',
												transform=data_transform)

	print('Number of train images: ', len(transformed_dataset))

	# iterate through the transformed dataset and print some stats about the first few samples
	for i in range(4):
		sample = transformed_dataset[i]
		print(i, sample['image'].size(), sample['keypoints'].size())

	test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
											root_dir='data/test/',
											transform=data_transform)

	print('Number of test images: ', len(test_dataset))
	
	return transformed_dataset, test_dataset


def create_dataloader(train_data, test_data):
	batch_size = 128
	train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=4)
	test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=True, num_workers=4)
	return train_loader, test_loader


def net_sample_output(net, test_loader, device):
	
	# iterate through the test dataset
	for i, sample in enumerate(test_loader):
		
		# get sample data: images and ground truth keypoints
		images = sample['image']
		key_pts = sample['keypoints']

		# convert images to FloatTensors
		images = images.type(torch.FloatTensor).cuda()

		# images, key_pts = images.to(device), key_pts.to(device)

		# forward pass to get net output
		output_pts = net(images)
		
		# reshape to batch_size x 68 x 2 pts
		output_pts = output_pts.view(output_pts.size()[0], 68, -1)
		
		# break after first image is tested
		if i == 0:
			return images, output_pts, key_pts


def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
	num_plot_columns = 5
	num_plot_rows = -(-batch_size//num_plot_columns) # take the ceiling
	_, ax = plt.subplots(num_plot_rows, num_plot_columns, figsize=(20,10))

	for i in range(batch_size):
		plot_row_index = i//num_plot_columns
		plot_column_index = i%num_plot_columns

		# un-transform the image data
		image = test_images[i].cpu().data   # get the image from it's wrapper
		image = image.numpy()   # convert to numpy array from a Tensor
		image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

		# un-transform the predicted key_pts data
		predicted_key_pts = test_outputs[i].cpu().data
		predicted_key_pts = predicted_key_pts.numpy()
		# undo normalization of keypoints  
		predicted_key_pts = predicted_key_pts*50.0+100
		
		# plot ground truth points for comparison, if they exist
		ground_truth_pts = None
		if gt_pts is not None:
			ground_truth_pts = gt_pts[i].cpu()       
			ground_truth_pts = ground_truth_pts*50.0+100

		ax[plot_row_index, plot_column_index].imshow(np.squeeze(image), cmap='gray')
		ax[plot_row_index, plot_column_index].scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
		# plot ground truth points as green pts
		if gt_pts is not None:
			ax[plot_row_index, plot_column_index].scatter(ground_truth_pts[:, 0], ground_truth_pts[:, 1], s=20, marker='.', c='g')
		
		ax[plot_row_index, plot_column_index].axis('off')

	plt.show()


def train_net(device, n_epochs, net, train_loader, test_loader, criterion, optimizer):
	train_losses = np.zeros(n_epochs)
	test_losses = np.zeros(n_epochs)
	net.train()

	for epoch in range(n_epochs):  # loop over the dataset multiple times
		train_loss = []
		t0 = datetime.now()

		# train on batches of data, assumes you already have train_loader
		for batch_i, data in enumerate(train_loader):
			images = data['image']
			key_pts = data['keypoints']

			# flatten pts
			key_pts = key_pts.view(key_pts.size(0), -1)

			# convert variables to floats for regression loss
			key_pts = key_pts.type(torch.FloatTensor).cuda()
			images = images.type(torch.FloatTensor).cuda()

			# move data to gpu
			# images, key_pts = images.to(device), key_pts.to(device)

			# forward pass to get outputs
			output_pts = net(images)

			# calculate the loss between predicted and target keypoints
			loss = criterion(output_pts, key_pts)

			# zero the parameter (weight) gradients
			optimizer.zero_grad()
			
			# backward pass to calculate the weight gradients
			loss.backward()

			# update the weights
			optimizer.step()

			train_loss.append(loss.item())

			torch.cuda.empty_cache()

		# save the train and test loss for this epoch
		train_loss = np.mean(train_loss) 
		
		net.eval()
		test_loss = []
		for data in test_loader:
			images = data['image']
			key_pts = data['keypoints']
			key_pts = key_pts.view(key_pts.size(0), -1)

			images = images.type(torch.FloatTensor).cuda()
			key_pts = key_pts.type(torch.FloatTensor).cuda()

			# images, key_pts = images.to(device), key_pts.to(device)

			outputs = net(images)
			loss = criterion(outputs, key_pts)
			test_loss.append(loss.item())
			torch.cuda.empty_cache()
		test_loss = np.mean(test_loss)

		# Save losses
		train_losses[epoch] = train_loss
		test_losses[epoch] = test_loss

		dt = datetime.now() - t0
		print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Duration: {dt}')

	print('Finished Training')
	return train_losses, test_losses


def main():
	# instantiate the network
	net = Net().cuda()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# print(device)
	# net.to(device)
	# print(net)

	train_data, test_data = get_train_test_data()
	train_loader, test_loader = create_dataloader(train_data, test_data)

	test_images, test_outputs, gt_pts = net_sample_output(net, test_loader, device)

	# print the dimensions of the data to see if they make sense
	print('test data dimensions:')
	print(test_images.data.size())
	print(test_outputs.data.size())
	print(gt_pts.size())
	# visualize_output(test_images, test_outputs, gt_pts)

	# define the loss and optimization
	learn_rate = 0.001
	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters(), lr=learn_rate)

	# train your network
	n_epochs = 500
	train_losses, test_losses = train_net(device, n_epochs, net, train_loader, test_loader, criterion, optimizer)

	model_dir = 'saved_models/'
	model_name = 'keypoints_model_2.pt'

	# after training, save your model parameters in the dir 'saved_models'
	torch.save(net.state_dict(), model_dir+model_name)

	plt.figure()
	plt.plot(train_losses, label='train loss')
	plt.plot(test_losses, label='test loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.grid()
	plt.show()

	test_images, test_outputs, gt_pts = net_sample_output(net, test_loader, device)
	visualize_output(test_images, test_outputs, gt_pts)

	del net
	torch.cuda.empty_cache()

if __name__ == '__main__':
	main()
