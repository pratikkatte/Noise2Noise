import torch
import torch.optim as optim
from dataset import *
from network import *
import argparse
import time


def createLossAndOptimizer(net, learning_rate):
    
    #Loss function
    loss = torch.nn.MSELoss()
    
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    return(loss, optimizer)


def train(net, batch_size, n_epochs, learning_rate,image_size, noise_model, dataset_path):
	print("===== HYPERPARAMETERS =====")
	print("batch_size=", batch_size)
	print("epochs=", n_epochs)
	print("learning_rate=", learning_rate)
	print("=" * 30)
	
	train_loader = getTrainLoader(dataset_path, noise_model, batch_size)

	test_loader = getLabledLoader(dataset_path,noise_model,batch_size)

	n_batches = len(train_loader)

	loss, optimizer = createLossAndOptimizer(net, learning_rate)

	training_start_time = time.time()

	for epoch in range(n_epochs):

		running_loss = 0.0
		print_every = n_batches // 10
		start_time = time.time()
		total_train_loss = 0

		for i, data in enumerate(zip(train_loader, test_loader)):


			optimizer.zero_grad()
			inputs, labels = data
			# inputs = inputs.float()
			# labels = labels.float()
			inputs = inputs.to(device)
			labels = labels.to(device)

			# print(inputs.shape)

			outputs = net(inputs)

			loss_size = loss(outputs, labels)
			loss_size.backward()
			optimizer.step()

			running_loss += loss_size.data[0]

			total_train_loss += loss_size.data[0]

			print(i,end='\r')

			if (i + 1) % (print_every + 1) == 0:
				print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
						epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))

				#Reset running loss and time

			running_loss = 0.0
			start_time = time.time()

def get_args():
    parser = argparse.ArgumentParser(description="Data set Prep for image restoration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_size", type=int, default=256,
                        help="training patch size")
    parser.add_argument("--noise_model", type=str, default="gaussian",
                        help="noise model to be trained")
    parser.add_argument("--path", type=str, default='/home/turing/Documents/BE/data', help="dataset folder path")

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
	
	num_epochs = 200
	batch_size = 4 
	learning_rate = 1e-3

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	tokens = get_args()

	image_size = tokens.image_size

	noise_model = tokens.noise_model

	dataset_path = tokens.path

	net = AutoEncoder().to(device)

	train(net, batch_size, num_epochs, learning_rate,image_size=image_size, noise_model=noise_model, dataset_path=dataset_path)

