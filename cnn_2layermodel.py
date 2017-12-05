from __future__ import print_function, division
import os, glob, argparse
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torch.autograd import Variable
from scipy.ndimage import imread
import shutil

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_EPOCHS = 20
DEFAULT_BATCH_SIZE = 100
DEFAULT_ROOT_DIR = 'set01'

class HangulDataset(Dataset):
    """Hangul Handwritten dataset."""

    def __init__(self, root_dir, subroot='char_data', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.subroot = subroot
        self.transform = transform
        self.classlist = os.listdir(os.path.join(root_dir, self.subroot))
        self.targets = []
        self.filenames = []
        self.targetdict = {}
        for i, label in enumerate(self.classlist):
            files = glob.glob(
                        os.path.join(self.root_dir, self.subroot, label) + '/*')
            self.filenames += files
            self.targets += [i] * len(files)
            self.targetdict[i] = int(label, 16)
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        target = self.targets[idx]
        sample = imread(img_name, mode='L')
        if self.transform:
            sample = self.transform(sample)
            sample = (sample - 0.97) / 0.09 
        return (sample, target)
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (self.output_size, self.output_size),
                               mode='reflect')

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        return img
    
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        dims = sample.shape
        if len(dims) == 2:
            image = np.expand_dims(sample, 0)
        else:
            image = sample.transpose((2, 0, 1))
        return torch.from_numpy(image)

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
                
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.fc = nn.Linear(93312, 2350)
        
    def forward(self, x):
        out = self.layer1(x.float())
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def show_image(image, cmap='gray'):
    """Show image"""
    plt.imshow(image, cmap=cmap)
    plt.show()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def test_model(val_loader, model, mode='Validate', use_gpu):
    # Test or validate the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in val_loader:
        if use_gpu:
                images = Variable(images.cuda(), volatile=True)
                labels = Variable(labels.cuda(), volatile=True)
        else:
                images = Variable(images, volatile=True)
                labels = Variable(labels, volatile=True)
        # compute output
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    
    acc = 100 * correct / total
    print('{0:%s} Accuracy: {1:.4f} %%'.format(mode, acc))
    return acc

def main(num_epochs, batch_size, learning_rate, root_dir):
    hangul_dataset = HangulDataset(root_dir, transform=Rescale(256))
    transformed_dataset = HangulDataset(root_dir='set01',
                                     transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True)

    # train, test data split
    num_data = len(hangul_dataset.filenames)
    indices = list(range(num_data))
    np.random.seed(42)
    np.random.shuffle(indices)

    val_size = 0.20
    test_size = 0.20
    test_split = int(np.floor(test_size * num_data))
    val_split = test_split + int(np.floor(val_size * num_data))
    num_train = num_data - val_split - test_split
    train_idx, val_idx, test_idx = indices[val_split:], indices[test_split:val_split] , indices[:test_split]

    # Hyper Parameters
#    num_epochs = 20
#    batch_size = 100
#    learning_rate = 0.001
   
     # Define sampler
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Train, test dataset loader
    train_loader = DataLoader(transformed_dataset, 
                            batch_size=batch_size, sampler=train_sampler)

    val_loader = DataLoader(transformed_dataset, 
                            batch_size=batch_size, sampler=val_sampler)

    test_loader = DataLoader(transformed_dataset, 
                            batch_size=batch_size, sampler=test_sampler)
   
     # use gpu if cuda is available
    use_gpu = torch.cuda.is_available()

    # Load model
    if use_gpu:
        cnn = CNN().cuda()
    else:
        cnn = CNN()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    
    loss_history = []
    best_acc = 0

    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if use_gpu:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.cpu().data.numpy())
            if i % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                       %(epoch+1, num_epochs, i, len(train_idx)//batch_size, loss.data[0]))

        # evaluate on validation set
        valacc = test_model(val_loader, cnn)

        # remember best prec@1 and save checkpoint
        is_best = valacc > best_acc
        best_acc = max(valacc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': cnn.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


    # Test the Model
    testacc = test_model(test_loader, cnn, mode='Test')

    # Save the Trained Model
    torch.save(cnn.state_dict(), 'cnn_2layers.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, dest='learning_rate',
                        default=DEFAULT_LEARNING_RATE,
                        help='Set learning rate')
    parser.add_argument('--num-epochs', type=int, dest='num_epochs',
                        default=DEFAULT_NUM_EPOCHS,
                        help='Set number of epochs')
    parser.add_argument('--batch-size', type=int, dest='batch_size',
                        default=DEFAULT_BATCH_SIZE,
                        help='Set batch size')
    parser.add_argument('--root-dir', type=str, dest='root_dir',
                        default=DEFAULT_ROOT_DIR,
                        help='Set root directory which containing image files. e.g. set01')
    args = parser.parse_args()
    main(args.num_epochs, args.batch_size, args.learning_rate, args.root_dir)
