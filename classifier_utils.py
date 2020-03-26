import os
import json
import torch
from torchvision import transforms
from torchvision import datasets
from torchvision import models
import torch.utils.data as data
from torch import optim, cuda
from torch import nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from PIL import Image
from collections import OrderedDict

plt.rcParams['font.size'] = 14

batch_size = 64

def get_data_dirs(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    dirs = {'train': train_dir, 'valid': valid_dir, 'test': test_dir}
    return dirs

def set_loaders(data_dir):
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    }
    
    image_datasets = {x: datasets.ImageFolder(data_dir[x], data_transforms[x]) for x in ['train', 'valid', 'test']}
    
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size = batch_size, shuffle = True) for x in ['train', 'valid', 'test']}
    
    return image_datasets, dataloaders

def get_model(arch):
    vgg16 = models.vgg16(pretrained=True)
    densenet161 = models.densenet161(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    _models = {'vgg': vgg16, 'densenet': densenet161, 'alexnet': alexnet}
    model = _models['vgg']
    
    if arch in _models:
        model = _models[arch]
    
    for param in model.parameters():
        param.requires_grad = False
    return model

def build_classifier(model, arch, hidden_sizes, dataloaders, drop_prob=0.5):
    print(">>> Building model classifier <<<")
    for param in model.parameters():
        param.requires_grad = False
    if arch== 'densenet':
        n_inputs = model.classifier.in_features
    else:
        n_inputs = model.classifier[-1].in_features
    
    n_classes = len(dataloaders['train'].dataset.classes)
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(n_inputs, hidden_sizes)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=drop_prob)),
        ('fc2', nn.Linear(int(hidden_sizes), n_classes)),
        ('out', nn.LogSoftmax(dim=1))
    ]))
    if arch== 'densenet':
        model.classifier = classifier
    else:
        model.classifier[-1] = classifier
    return model

def test_validation(model, valid_dataloaders, criterion, train_on_gpu):
    model.eval()
    valid_loss = 0
    valid_accuracy = 0
    
    with torch.no_grad():
        for images, labels in valid_dataloaders:
            if train_on_gpu:
                images, labels = images.cuda(), labels.cuda()
            
            output = model(images)
            
            loss = criterion(output, labels)
            
            valid_loss += loss.item() * images.size(0)
            
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(labels.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            valid_accuracy += accuracy.item() * images.size(0)
            
    valid_loss = valid_loss / len(valid_dataloaders.dataset)
    
    valid_accuracy = valid_accuracy / len(valid_dataloaders.dataset)
    return valid_loss, valid_accuracy

def train_model(model, learning_rate, train_loader, valid_loader, train_on_gpu, filename, epochs, epochs_stop=5, print_every=5):
    if train_on_gpu:
        model = model.to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=learning_rate)
    history = []
    model.epochs = 0
    epochs_without_improve = 0
    valid_loss_min = np.Inf
    
    start_time = timer()
    
    for epoch in range(epochs):
        print('Training started.....')
        print('Device is GPU?....{}'.format(train_on_gpu))
        train_loss = 0.0
        train_acc = 0
        model.train()

        for ii, (images, labels) in enumerate(train_loader):
            if train_on_gpu:
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, pred = torch.max(output, dim = 1)
            correct_tensor = pred.eq(labels.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            train_acc += accuracy.item() * images.size(0)
        else:
            model.epochs += 1
            model.eval()
            
            with torch.no_grad():
                valid_loss, valid_acc = test_validation(model, valid_loader, criterion, train_on_gpu)
                train_loss = train_loss / len(train_loader.dataset)
                train_acc = train_acc / len(train_loader.dataset)
                history.append([train_loss, valid_loss, train_acc, valid_acc])
            
                if (epoch + 1) % print_every == 0:
                    print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Training Loss: {:.3f}.. ".format(train_loss),
                      "Valid Loss: {:.3f}.. ".format(valid_loss),
                      "Train Accuracy: {:.2f}%".format(100 * train_acc),
                      "Valid Accuracy: {:.2f}%".format( 100 * valid_acc))
           
                if valid_loss < valid_loss_min - 0.01:
                    torch.save(model.state_dict(), filename)
                    epochs_without_improve = 0
                    valid_loss_min = valid_loss
                    best_epoch = epoch
                else:
                    epochs_without_improve += 1
                    if epochs_without_improve >= epochs_stop:
                        total_time = timer() - start_time
                        print('Total time: {:.2f} seconds.Time/epoch: {:.2f} seconds'.format(total_time, total_time / (epoch+1)))
                        model.load_state_dict(torch.load(filename))
                        model.optimizer = optimizer
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history
                
    model.optimizer = optimizer
    total_time = timer() - start_time
    print('The best epoch: {} with loss {:.2f} and accuracy: {:.2f}%'.format(best_epoch, valid_loss_min,100* valid_acc))
    if epoch != 0:
        print("Total time: {:.2f} seconds. Time per epoche was {:.2f} seconds".format(total_time, total_time / epoch ))
    history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history

def test_model(model, testloader, train_on_gpu):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = test_validation(model, testloader, criterion, train_on_gpu)
    print("Test Loss: {:.3f}.. ".format(test_loss),"Accuracy: {:.2f}%".format( 100 * test_acc))
    
def idx_to_name(train_set):
    cat_name = cat_to_name()
    idx_to_name = {
        idx: cat_name[class_]
        for class_, idx in train_set.class_to_idx.items()
    }
    return idx_to_name
    
def cat_to_name():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name

def save_checkpoint(model, epochs, learning_rate, path, train_set):
    print('Saving the model to ./{}/checkpoint.pth'.format(path), flush = True)
    checkpoint = {
        'model': model,
        'cat_to_name': cat_to_name(),
        'class_to_idx': train_set.class_to_idx,
        'idx_to_name': idx_to_name(train_set),
        'epochs': epochs,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer': model.optimizer.state_dict()
    }
    
    if not os.path.exists(path):
        print('save directories...', flush = True)
        os.makedirs(path)
    torch.save(checkpoint, path + '/checkpoint.pth')
    
def load_checkpoint(path):  
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_name = checkpoint['idx_to_name']
    model.classifier = checkpoint['classifier']
    model.cat_to_name = checkpoint['cat_to_name']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    width, height = img.size
    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    width, height = img.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    
    img = np.array(img)
    img = img.transpose((2, 0, 1))
    img = img/255
    
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    img = img[np.newaxis,:]
    
    image = torch.from_numpy(img)
    image = image.float()
    return image

def predict(image, model, train_on_gpu,topk=5):
    if train_on_gpu:
        model = model.to('cuda')
        image = image.view(1, 3, 224, 224).cuda()
    else:
        image = image.view(1, 3, 224, 224)
    with torch.no_grad():    
        model.eval()
        out = model(image)   
        ps = torch.exp(out)
        topk, topclass = ps.topk(topk, dim=1)
        top_classes = [
            model.idx_to_name[category] for category in topclass.cpu().numpy()[0]
        ]
        top_prob = topk.cpu().numpy()[0]
        
        return top_prob, top_classes, topclass.cpu().numpy()[0]
      