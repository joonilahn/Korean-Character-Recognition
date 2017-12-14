from __future__ import print_function, division
import os, glob, argparse, shutil, time, json
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils, models
from torch.autograd import Variable
from torch.optim import lr_scheduler
from scipy.ndimage import imread
import cv2
from logger import Logger
from datetime import datetime

DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_NUM_EPOCHS = 20
DEFAULT_BATCH_SIZE = 128
DEFAULT_ROOT_DIR = 'set01'
DEFAULT_NUM_CLASSES = 2350
DEFAULT_USE_MODEL = 'vgg19'

# use gpu if cuda is available
use_gpu = torch.cuda.is_available()

# CNN Model (2 conv layer)
class Baseline(nn.Module):
    def __init__(self, num_classes):
        super(Baseline, self).__init__()
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
        
        self.classifier = nn.Sequential(
            nn.Linear(93312, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x.float())
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

# Custom VGG19_bn
class CustomVGG19bn(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG19bn, self).__init__()
        self.features = nn.Sequential(*list(models.vgg19_bn(pretrained=True).features.children()))
        self.classifier = nn.Sequential(
            *[list(models.vgg19_bn(pretrained=True).classifier.children())[i] for i in range(6)],
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512*7*7)
        out = self.classifier(x)
        return out

# Custom resnet50
class CustomResnet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResnet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:9])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 2048)
        out = self.classifier(x)
        return out

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
        height, width = sample.shape[0], sample.shape[1]
        recrop = 0
        # Find contours
        (_, contours, _) = cv2.findContours(sample.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # if no contour found
        if len(contours) == 0:
            return sample

        # one contour found
        if len(contours) == 1:
            [x, y, w, h] = cv2.boundingRect(contours[0])
            yh = y + h
            xw = x + w

        # multiple contours found
        else:
            for i, contour in enumerate(contours):
                if i==0:
                    [x, y, w, h] = cv2.boundingRect(contour)
                    yh = y + h
                    xw = x + w
                else:
                    [new_x, new_y, new_w, new_h] = cv2.boundingRect(contour)
                    # Ignore too narrow contours (outliers)
                    if new_w < 5or new_h < 5:
                        continue
                    new_yh = new_y + new_h
                    new_xw = new_x + new_w
                    if new_x < x:
                        x = new_x
                    if new_y < y:
                        y = new_y
                    if new_yh > yh: 
                        yh = new_yh
                    if new_xw > xw:
                        xw = new_xw

        if yh-y < 50 or xw-x < 50:
            recrop = 1
        else:
            if yh-y > height*0.97 and xw-x > width*0.97:
                recrop = 1
            else:
                sample = sample[y:yh, x:xw]

        if recrop == 0:
            return sample
        else:
            maxpix = sample.max()
            brightidx = np.where(sample > maxpix*0.8)
            min_h, min_w = np.min(brightidx, axis=1)
            max_h, max_w = np.max(brightidx, axis=1)
            sample = sample[min_h:max_h, min_w:max_w]
            return sample

class Denoise(object):
    """Stablize pixel values of images."""
    def __call__(self, sample):
        if sample.min() > 200:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            sample = clahe.apply(sample)
        sample = cv2.threshold(sample, 225, 255, cv2.THRESH_BINARY_INV)[1]
        return sample
    
class Normalize(object):
    """Normalize images."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        return (sample - self.mean) / self.std

# train model function
def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # Set the logger
    now = datetime.now()
    logdir = "./logs/" + now.strftime("%Y%m%d-%H%M%S")
    logger = Logger(logdir)

    for epoch in range(num_epochs):
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
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    if phase == 'train':
                        inputs = Variable(inputs).float().cuda()
                        labels = Variable(labels).long().cuda()
                    else:
                        inputs = Variable(inputs, volatile=True).float().cuda()
                        labels = Variable(labels, volatile=True).long().cuda()
                else:
                    if phase == 'train':
                        inputs, labels = Variable(inputs).float(), Variable(labels).long()
                    else:
                        inputs, labels = Variable(inputs, volatile=True).float(), Variable(labels, volatile=True).long()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                running_total += len(labels.data)

                # print every 10 iterations
                if (i+1) % 100 == 0:
                    print('Epoch: {0:}/{1:}, Iterations: {2:}/{3:}, {4:} loss: {5:6.2f}'.
                     format(epoch+1, num_epochs, i+1, len(dataloaders[phase]), phase, loss.data[0]))

                if i % 1000 == 0:
                    #============ TensorBoard logging ============#
                    # Save log file every 1000 iteration
                    # (1) Log the scalar values
                    if phase == 'train':
                        info = {
                            'epoch': epoch,
                            'train loss': loss.data[0],
                            'train accuracy': running_corrects/running_total
                        }
                        for tag, value in info.items():
                            logger.scalar_summary(tag, value, i+1)
                    else:
                        info = {
                            'epoch': epoch,
                            'validation loss': loss.data[0],
                            'validation accuracy': running_corrects/running_total
                        }
                        for tag, value in info.items():
                            logger.scalar_summary(tag, value, i+1)
                    # # (2) Log values and gradients of the parameters (histogram)
                    # for tag, value in model.named_parameters():
                    #     tag = tag.replace('.', '/')
                    #     logger.histo_summary(tag, value.data.cpu().numpy(), i+1)
                    #     logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), i+1)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / running_total

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, 'bestmodel.pt')
        # save checkpoint
        save_model(model, optimizer, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, testloader):
    if use_gpu:
        model = model.cuda()
    model.eval()
    correct = 0
    total = 0
    print('Testing the model')
    for i, data in enumerate(testloader):
        images, labels = data
        if use_gpu:
            images = Variable(images, volatile=True).cuda()
            labels = labels.cuda()
        else:
            images = Variable(images, volatile=True).float()
        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += torch.sum(preds == labels)
        if (i+1) % 100 == 0:
            print('Test iterations: {0:}/{1:}'.format( (i+1), len(testloader)))

    acc = correct / total
    print('Test Accuracy of the model on the %d test images: %.4f' % (total, acc))
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

def main(num_epochs, batch_size, learning_rate, root_dir, num_classes, use_model):
    # load model
    if use_model == "vgg19":
        model = CustomVGG19bn(num_classes=num_classes)
    if use_model == "resnet50":
        model = CustomResnet50(num_classes=num_classes)
    if use_model == "baseline":
        model = Baseline(num_classes=num_classes)

    # Load dataset
    if use_model == "baseline":
        transformed_dataset = HangulDataset(root_dir=root_dir,
                                    transform=transforms.Compose([
                                    Denoise(),
                                    ObjectCrop(),
                                    Normalize(50.81, 110.16),
                                    Rescale(224),
                                    ToTensor(),  
                                           ]),
                                    num_class=num_classes)
    else: 
        transformed_dataset = HangulDataset(root_dir=root_dir,
                                    transform=transforms.Compose([
                                    Denoise(),
                                    ObjectCrop(),
                                    Normalize(50.81, 110.16),
                                    Rescale(224),
                                    ToTensor(),  
                                    transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                           ]),
                                    num_class=num_classes)

    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze parameters in classification layer
    for layer_idx, param in enumerate(model.classifier.parameters()):
        param.requires_grad = True
    
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

    # Hyper Parameters
    # num_epochs = 20
    # batch_size = 100
    # learning_rate = 0.001

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
    
    dataloaders = {'train':train_loader, 'val':val_loader}

    if use_gpu:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.05)

    # Train the model
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders,
                       num_epochs=num_epochs)
    # model.load_state_dict(torch.load('bestmodel.pt'))
    # Test the model
    testacc = test_model(model, test_loader)

    # Write result in json file
    resultdict = {'Model':use_model, 'Test_accuracy':testacc, 'Learning_rate':learning_rate,
                    'Epochs':num_epochs, 'Batch_size':batch_size, 'Num_classes':num_classes 'Datasets':root_dir}
    now = datetime.now()
    jsonfile = "result_" + now.strftime("%Y%m%d-%H%M%S") + ".json"
    with open(jsonfile, 'w') as f:
        json.dump(resultdict, f)

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
    parser.add_argument('--use-model', type=str, dest='use_model',
                        default=DEFAULT_USE_MODEL,
                        help='Set which model to be used for the training e.g. vgg19 or resnet50')
    args = parser.parse_args()
    main(args.num_epochs, args.batch_size, args.learning_rate, args.root_dir, args.num_classes, args.use_model)
