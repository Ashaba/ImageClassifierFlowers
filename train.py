import sys
import argparse
from classifier_utils import *
from workspace_utils import active_session

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type = str, default = 'flowers', help = 'images data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help = 'directory for for saving checkpoints')
    parser.add_argument('--arch', type=str, default = 'vgg13', help = 'Choose architecture')
    parser.add_argument('--learning_rate',  type=float, default=0.01, help = 'Learning rate')  
    parser.add_argument('--hidden_units', type=int, default=512, help= 'Set the number of hidden units')
    parser.add_argument('--epochs',  type=int, default=10, help='set the number of epochs to train the model')
    parser.add_argument('--gpu',  action='store_true', help = 'set the train on gpu')  
 
    return parser.parse_args()

def main():
    args = get_input_args()
    is_device_gpu = False
    filename = 'flowers_classifier.pt'
    if args.gpu:
        is_device_gpu = cuda.is_available() 
    data_dir = get_data_dirs(args.data_dir)
    image_datasets, dataloaders  = set_loaders(data_dir)
    model = get_model(args.arch)
    model = build_classifier(model, args.arch,args.hidden_units, dataloaders)

    train_model(model, args.learning_rate, dataloaders['train'], dataloaders['valid'], is_device_gpu, filename, args.epochs)   
    test_model(model, dataloaders['test'], is_device_gpu)
    save_checkpoint(model, args.epochs,args.learning_rate ,args.save_dir, image_datasets['train'])
    print('\n >>> Training Completed! <<< \n', flush = True)
    
if __name__ == "__main__":
    sys.stdout.flush()
    with active_session():
        main()