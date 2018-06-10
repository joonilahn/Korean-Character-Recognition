from __future__ import print_function, division
import os, glob, argparse, json
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils, models
import torch.nn.functional as F
from torchvision.models.inception import BasicConv2d, InceptionA, InceptionB, \
                                        InceptionC, InceptionD, InceptionE

from skimage import io, transform
from scipy.ndimage import imread, median_filter
import cv2
import shutil
import time
from logger import Logger
from datetime import datetime

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_ROOT_DIR = 'all'
DEFAULT_NUM_CLASSES = 2350

# use gpu if cuda is available
use_gpu = torch.cuda.is_available()

# Custom VGG19_bn
class inception_v3_1c(nn.Module):
    def __init__(self, num_classes=2350, aux_logits=True, transform_input=False):
        super(inception_v3_1c, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5

        # 149 x 149 x 1
        x = self.Conv2d_1a_3x3(x)
        
        # 74 x 74 x 32
        x = self.Conv2d_2a_3x3(x)
        
        # 72 x 72 x 32
        x = self.Conv2d_2b_3x3(x)
        
        # 72 x 72 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        # 35 x 35 x 64
        x = self.Conv2d_3b_1x1(x)
        
        # 35 x 35 x 80
        x = self.Conv2d_4a_3x3(x)
        
        # 33 x 33 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        # 16 x 16 x 192
        x = self.Mixed_5b(x)
        
        # 16 x 16 x 256
        x = self.Mixed_5c(x)
        
        # 16 x 16 x 288
        x = self.Mixed_5d(x)
        
        # 16 x 16 x 288
        x = self.Mixed_6a(x)
        
        # 7 x 7 x 768
        x = self.Mixed_6b(x)
        
        # 7 x 7 x 768
        x = self.Mixed_6c(x)
        
        # 17 x 17 x 768
        # 7 x 7 x 768
        x = self.Mixed_6d(x)
        
        # 7 x 7 x 768
        x = self.Mixed_6e(x)
        
        # 7 x 7 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        
        # 7 x 7 x 768
        x = self.Mixed_7a(x)
        
        # 3 x 3 x 1280
        x = self.Mixed_7b(x)
        
        # 3 x 3 x 2048
        x = self.Mixed_7c(x)
        
        # 3 x 3 x 2048
        x = F.avg_pool2d(x, kernel_size=3)
        
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        
        # 2048
        x = self.fc(x)
        
        # 2350 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x

class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes=2350):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=3)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 7 x 7 x 768
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # 3 x 3 x 768
        x = self.conv0(x)
        # 3 x 3 x 128
        x = self.conv1(x)  
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 2350
        return x

class HangulDataset(Dataset):
    """Hangul Handwritten dataset."""

    def __init__(self, root_dir, subroot='char_data', transform=None, num_class=2350):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if root_dir == 'all':
            self.root_dir = ['set01', 'set02', 'set03', 'set04', 'set05',
                                  'set06', 'set07', 'set08', 'set09']
        else:
            self.root_dir = root_dir
        self.subroot = subroot
        self.transform = transform
        self.num_class = num_class
        # Build classlist
        if root_dir == 'all':
            self.classlist = os.listdir(os.path.join(self.root_dir[0], self.subroot))
            if self.num_class != 2350:
                with open('256hangul.json', 'r') as f:
                    class256 = json.load(f)
                class256 = list(class256.keys())
                class256 = [int(x) for x in class256]
                new_classlist = []
                for cl in self.classlist:
                    if int(cl, 16) in class256:
                        new_classlist.append(cl)
                self.classlist = new_classlist
        else:
            self.classlist = os.listdir(os.path.join(self.root_dir, self.subroot))
            if self.num_class != 2350:
                with open('256hangul.json', 'r') as f:
                    class256 = json.load(f)
                class256 = list(class256.keys())
                class256 = [int(x) for x in class256]
                new_classlist = []
                for cl in self.classlist:
                    if int(cl, 16) in class256:
                        new_classlist.append(cl)
                self.classlist = new_classlist
                
        self.targets = []
        self.filenames = []
        self.targetdict = {}
        for i, label in enumerate(self.classlist):
            if root_dir == 'all':
                files = []
                for rootdir in self.root_dir:
                    files += glob.glob(
                        os.path.join(rootdir, self.subroot, label) + '/*')
            else:
                files = glob.glob(
                        os.path.join(self.root_dir, self.subroot, label) + '/*')
            self.filenames += files
            self.targets += [i] * len(files)
            self.targetdict[i] = chr( int(label, 16) )
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        target = self.targets[idx]
        sample = imread(img_name, mode='L')
        if self.transform:
            sample = self.transform(sample)
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
        # h, w = image.shape[:2]
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        #     new_h, new_w = self.output_size

        # new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (self.output_size, self.output_size),
                               mode='reflect')
        return img
    
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
        sampletensor = torch.from_numpy(image)
        return sampletensor.type(torch.FloatTensor)
    
class ObjectCrop(object):
    """Use findContours function of OpenCV to detect object and crop the images"""            
    def __call__(self, sample):
        maxpix = sample.max()
        brightidx = np.where(sample > maxpix*0.9)

        if np.sum(brightidx) == 0:
            return sample

        else:
            brightidx = np.where(sample > maxpix*0.9)
            min_h, min_w = np.min(brightidx, axis=1)
            max_h, max_w = np.max(brightidx, axis=1)
            if max_h - min_h < 10 or max_w - min_w < 10:
                return sample
            else:
                return sample[min_h:max_h, min_w:max_w]

class Denoise(object):
    """Stablize pixel values of images."""
    def __call__(self, sample):
        if sample.min() > 160:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            sample = clahe.apply(sample)
            sample = cv2.threshold(sample, 254, 255, cv2.THRESH_BINARY_INV)[1]
            sample = median_filter(sample, 5)
            return sample
        sample = cv2.threshold(sample, 225, 255, cv2.THRESH_BINARY_INV)[1]
        sample = median_filter(sample, 5)
        return sample
    
class Normalize(object):
    """Normalize images."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        return (sample - self.mean) / self.std

# train model function
def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25, print_every=100, start_epoch=0):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    # Set the logger
    now = datetime.now()
    logdir = "./logs/" + now.strftime("%Y%m%d-%H%M%S")
    logger = Logger(logdir)
    global_iterations = 0

    for epoch in range(num_epochs):
        epoch += start_epoch

        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_total = 0
            
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                global_iterations += 1

                if i % print_every == 0:
                    i_start_time = time.time()

                # get the inputs
                inputs, labels = data

                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                if phase == 'train':
                    outputs, aux_output = model(inputs)
                else:
                    outputs = model(inputs)

                # prediction
                _, preds = torch.max(outputs.data, 1)

                # compute loss
                loss_1 = criterion(outputs, labels)
                if phase == 'train':
                    loss_aux = criterion(aux_output, labels)
                    loss = loss_1 + loss_aux
                else:
                    loss = loss_1

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()
                running_total += len(labels.data)
                
                # print every 100 iterations
                if (i+1) % print_every == 0:
                    print('Epoch: {0:}/{1:}, Iterations: {2:}/{3:}, {4:} loss: {5:6.2f}'.
                         format(epoch+1, num_epochs, i+1, len(dataloaders[phase]), phase, running_loss / (i+1)))
                    # print time
                    time_elapsed = time.time() - i_start_time
                    print('It took {0:.0f}m {1:.0f}s for {2:d} iterations'.format(
                                    time_elapsed // 60, time_elapsed % 60, print_every))

                if i % 1000 == 0:
                    #============ TensorBoard logging ============#
                    # Save log file every 1000 iteration
                    # (1) Log the scalar values
                    if phase == 'train':
                        info = {
                            'epoch': epoch,
                            'train loss': loss.item(),
                            'train accuracy': running_corrects/running_total
                        }
                        for tag, value in info.items():
                            logger.scalar_summary(tag, value, global_iterations)
                    else:
                        info = {
                            'epoch': epoch,
                            'validation loss': loss.item(),
                            'validation accuracy': running_corrects/running_total
                        }
                        for tag, value in info.items():
                            logger.scalar_summary(tag, value, global_iterations)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / running_total * 100
#             epoch_acc = running_corrects / len(dataloaders[phase])

            print('{0:} Loss: {1:.4f} {2:} Acc: {3:.2f}%'.format(
                phase, epoch_loss, phase, epoch_acc))

            # tensorboard logging for every epoch
            if phase == 'train':
                logger.scalar_summary('Epoch train loss', epoch_loss, epoch+1)
                logger.scalar_summary('Epoch train accuracy', epoch_acc, epoch+1)
            else:
                logger.scalar_summary('Epoch validation loss', epoch_loss, epoch+1)
                logger.scalar_summary('Epoch validation accuracy', epoch_acc, epoch+1)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                save_model(model, optimizer, epoch, filename='best_model.pth.tar')
        save_model(model, optimizer, epoch, filename='checkpoint.pth.tar')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:2f}%'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, testloader, print_every=100):
    if use_gpu:
        model = model.cuda()
    model.eval()
    running_corrects = 0
    # total = 0
    print('Testing the model')
    for i, data in enumerate(testloader):
        images, labels = data
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)
        # total += labels.size(0)
        # correct += torch.sum(preds == labels).item()
        running_corrects += torch.sum(preds == labels.data).item()
        # running_total += len(labels.data)

        if (i+1) % print_every == 0:
            print('Test iterations: {0:}/{1:}'.format( (i+1), len(testloader)))

    acc = running_corrects / len(testloader) * 100
    print('Test Accuracy of the model on the {:d} test images: {:.2f}'.format(len(testloader), acc))
    return acc

def save_model(model, optimizer, epoch, filename='checkpoint.pth.tar'):
    state_dict = model.state_dict()                                                                                                                                                                         
    for key in state_dict.keys():                                                                                                                                                                                
        state_dict[key] = state_dict[key].cpu()                                                                                                                                                                  
                                                                                                                                                                                                                 
    torch.save({                                                                                                                                                                                                 
            'epoch': epoch,                                                                                                                                                                                     
            'state_dict': state_dict,                                                                                                                                                                                
            'optimizer': optimizer},                                                                                                                                                                                     
            filename)

def main(num_epochs, batch_size, learning_rate, root_dir, num_classes, resume_train):
    # load model
    model = inception_v3_1c(num_classes=num_classes)
    transformed_dataset = HangulDataset(root_dir=root_dir,
                                transform=transforms.Compose([
                                Denoise(),
                                ObjectCrop(),
                                Rescale(149),
                                ToTensor()
                                ]),
                                num_class=num_classes)

    # train, test data split
    num_data = len(transformed_dataset)
    indices = list(range(num_data))
    np.random.seed(42)
    np.random.shuffle(indices)

    val_size = 0.20
    test_size = 0.20
    test_split = int(np.floor(test_size * num_data))
    val_split = test_split + int(np.floor(val_size * num_data))
    num_train = num_data - val_split - test_split
    train_idx, val_idx, test_idx = indices[val_split:], indices[test_split:val_split] , indices[:test_split]

     # Define sampler
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Train, test dataset loader
    train_loader = DataLoader(transformed_dataset, 
                            batch_size=batch_size, sampler=train_sampler)

    val_loader = DataLoader(transformed_dataset, 
                            batch_size=batch_size, sampler=val_sampler)
    
    dataloaders = {'train':train_loader, 'val':val_loader}

    if use_gpu:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Decay LR by a factor of 0.1 every 7 epochs
    lr_scheldule = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    # lr_scheldule = lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5, 8], gamma=0.1)
    start_epoch = 0

    # Train the model
    if resume_train == True:
        ckpt = torch.load('checkpoint.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        optimizer = ckpt['optimizer']
        start_epoch = ckpt['epoch']

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = train_model(model, criterion, optimizer, lr_scheldule, dataloaders,
                                num_epochs=num_epochs, start_epoch=start_epoch)
    # model.load_state_dict(torch.load('bestmodel.pt'))

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
    parser.add_argument('--num-classes', type=int, dest='num_classes',
                        default=DEFAULT_NUM_CLASSES,
                        help='Set number of classes')
    parser.add_argument('--resume', type=bool, dest='resume_train',
                        default=False,
                        help='Resume train using a saved checkpoint')
    args = parser.parse_args()
    main(args.num_epochs, args.batch_size, args.learning_rate, args.root_dir, args.num_classes, args.resume_train)
