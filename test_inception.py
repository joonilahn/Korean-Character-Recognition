import numpy as np
from train_inception import inception_v3_1c, HangulDataset,\
                 Denoise, ObjectCrop, Rescale, ToTensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torchvision import transforms
from datetime import datetime
import json

# use gpu if cuda is available
use_gpu = torch.cuda.is_available()

def test_model(model, testloader, batch_size, print_every=100):
    if use_gpu:
        model = model.cuda()
    model.eval()
    running_corrects = 0
    running_total = 0
    
    print('Testing the model')
    for i, data in enumerate(testloader):
        images, labels = data
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)
        
        running_corrects += torch.sum(preds == labels.data).item()
        running_total += len(labels.data)

        if (i+1) % print_every == 0:
            print('Test iterations: {0:}/{1:}'.format( (i+1), len(testloader)))

    acc = running_corrects / running_total * 100
    print('Test Accuracy of the model on the {0:d} test images: {1:.2f} %'\
                                .format(len(testloader) * batch_size, acc))
    return acc

def main(root_dir, batch_size=128, num_classes=2350):
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

    # val_size = 0.20
    test_size = 0.20
    test_split = int(np.floor(test_size * num_data))
    # val_split = test_split + int(np.floor(val_size * num_data))
    # num_train = num_data - val_split - test_split
    # train_idx, val_idx, test_idx = indices[val_split:], indices[test_split:val_split] , indices[:test_split]
    test_idx = indices[:test_split]

     # Define sampler
    # train_sampler = SubsetRandomSampler(train_idx)
    # val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    if use_gpu:
        model = model.cuda()

    # load checkpoint for the best model
    ckpt = torch.load('best_model.pth.tar')
    best_model_wts = ckpt['state_dict']
    model.load_state_dict(best_model_wts)    

    # Test the model
    test_loader = DataLoader(transformed_dataset, 
                            batch_size=batch_size, sampler=test_sampler)
    testacc = test_model(model, test_loader, batch_size)

    # Write result in json file
    resultdict = {'Model':'Inception_v4', 'Test_accuracy':testacc, 'Datasets':root_dir}
    now = datetime.now()
    jsonfile = "result_" + now.strftime("%Y%m%d-%H%M%S") + ".json"
    with open(jsonfile, 'w') as f:
        json.dump(resultdict, f)

if __name__ == '__main__':
    main(root_dir='all')