0.Download dataset Fer-2013

1. Install packages:
	pip install torch
	pip install torchvision

2. Change the dataset path to your own

train_path = '/Users/john/Desktop/ML/Fer2013/train'  # for training
test_path = '/Users/john/Desktop/ML/Fer2013/test'   # for testing

3. Change important parameters, for example,
	epoch_times = 20						   # Total epochs
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1).           # Initial Learning rate
	scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # every 5times, LR  will decay 0.1 times

4. Run all and wait for the results. Because this dataset is very big, so each epoch runs slow. However, you can know the process though print function, which is at bottom of the code. After the training and testing, a figure will come. 

