import argparse
import os, sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datetime import datetime as dtm
import pandas as pd
import numpy as np
import shutil
from logger import Logger
from custom_scheduler import ReduceLROnPlateau
from model_baseline import DNN, CNN
from getter_dataset import MfccLoader
from getter_dataset_cnn import MfccLoaderCNN


# support function
def save_checkpoint(state, filename, is_best):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'best_' + os.path.basename(filename)))

def to_np(x):
    return x.data.cpu().numpy()

# logging
def logging(loss, running_acc, total, is_train, step, is_per_epoch, inputs=None):
    #============ TensorBoard logging ============#
    # (1) Log the scalar values
    accuracy = 100.0 * running_acc / total
    loss_str = 'Loss per epoch' if is_per_epoch else 'Loss per step' 
    accuracy_str = 'Accuracy per epoch' if is_per_epoch else 'Accuracy per step'
    info = {
        loss_str: loss,
        accuracy_str: accuracy
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, step, is_train)

    # (2) Log values and gradients of the parameters (histogram)
    #for tag, value in filter(lambda p: p[1].requires_grad, model.named_parameters()):
    #    tag = tag.replace('.', '/')
    #    logger.histo_summary(tag, to_np(value), step, 1000, is_train)
    #    logger.histo_summary(tag + '/grad', to_np(value.grad), step, 1000, is_train)

# parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model.')

    parser.add_argument('--model',
                        help='model')

    parser.add_argument('--model_seq',
                        help='model sequence')

    parser.add_argument('--model_path',
                        nargs=1,
                        required=True,
                        help='path to model\'s file')

    parser.add_argument('--model_name',
                        nargs=1,
                        required=True,
                        help='Filename with model')

    parser.add_argument('--batch_size',
                        nargs=1,
                        default=[1],
                        type=int,
                        help='size of batch')

    parser.add_argument('--list_postfix',
                        nargs=1,
                        required=True,
                        help='list filename postfix')

    parser.add_argument('--list_path',
                        nargs=1,
                        required=True,
                        help='path to file with list')

    parser.add_argument('--epochs',
                        nargs=1,
                        type=int,
                        required=True,
                        help='number of epochs')

    parser.add_argument('--optimizer',
                        nargs=1,
                        default=['adam'],
                        help='optimization method for model\'s train')

    parser.add_argument('--lr',
                        nargs=1,
                        type=float,
                        default=[0.1],
                        help='learning rate')

    parser.add_argument('--resume',
                        nargs=1,
                        type=str,
                        help='file with checkpoint information')

    parser.add_argument('--checkpoint',
                        nargs=1,
                        type=str,
                        help='file for writing checkpoint information')

    parser.add_argument('--logger_dir',
                        nargs=1,
                        type=str,
                        required=True,
                        help='path to log files')

    parser.add_argument('--wd',
                        nargs=1,
                        type=float,
                        required=True,
                        help='weight decay')

    parser.add_argument('--data_path',
                        nargs=1,
                        required=True,
                        help='data path')


    args = parser.parse_args()

    # getting list filename
    list_postfix = args.list_postfix[0]

    # getting list path
    list_path = args.list_path[0]

    # getting batch size
    batch_size = args.batch_size[0]

    # getting number of epochs
    number_epochs = args.epochs[0]

    # getting out filename
    out_filename = os.path.join(args.model_path[0], args.model_name[0])

    # getting data path
    data_path = args.data_path[0]

    # getting optimizer
    optimizers = {
        'sgd'   : optim.SGD,
        'adam'  : optim.Adam,
    }
    optimizer = optimizers[args.optimizer[0]]

    # getting learning rate
    lr = args.lr[0]

    # getting weight decay
    weight_decay = args.wd[0]

    # getting train/val filenames
    list_train = os.path.join(list_path, 'train_list_' + list_postfix)
    list_val = os.path.join(list_path, 'val_list_' + list_postfix)


    # getting model
    if args.model and args.model == 'CNN':
        model = CNN(num_classes=3)
        Loader = MfccLoaderCNN
    else:
        model_seq = args.model_seq.split(',')
        model_seq = [int(i) for i in model_seq]
        model = DNN(model_seq)
        Loader = MfccLoader
    model.cuda()

    # getting checkpoint
    checkpoint_filename = os.path.join(args.model_path[0], args.checkpoint[0])

    # getting log directory
    logger_dir = args.logger_dir[0]

    print(sys.argv)
    #print("Model DNN: ", model_seq)
    print("Train list file: ", list_train)
    print("Validation list file: ", list_val)
    print("Size of batch: ", batch_size)
    print("Number of epochs: ", number_epochs)
    print("Out model file: ", out_filename)
    print("Optimizer: ", args.optimizer[0])
    print("Learning rate: ", args.lr[0])
    print("Checkpoint file for saving: ", checkpoint_filename)
    if args.resume:
        print("Checkpoint file for resuming: ", args.resume[0])
    print("Model arch:")
    print(model)
    
    means = (-3.06361016e+02,  1.05540514e+02, -6.81725492e+00,  3.52314962e+01,
        -8.43053763e-02,  1.31217474e+01, -6.32619909e+00,  5.97222811e+00,
        -1.00541418e+01,  5.30146802e+00, -5.15876793e+00,  5.44507806e+00,
            9.91734609e-03,  9.42358188e-04, -1.37527333e-03,  8.39388125e-04,
        -9.11329246e-04,  9.99707518e-04, -1.14366317e-03,  7.51635081e-04,
        -7.99993116e-04,  2.35506141e-04, -8.05091586e-04,  8.39820745e-04,
            1.74729401e-04, -1.81443088e-03,  5.89198023e-04, -1.07557727e-03,
            1.10021589e-03, -7.69683687e-04,  4.08600170e-04, -6.53653393e-04,
            6.47737162e-04, -2.97134126e-04,  5.81508539e-04, -4.07872409e-04,
            1.26295894e+05,  1.53637724e+02,  5.40922254e+01)
    stds = (1.17794075e+02, 4.58205444e+01, 3.27203664e+01, 2.42906558e+01,
        1.95046469e+01, 1.82702450e+01, 1.61740334e+01, 1.58938929e+01,
        1.40754715e+01, 1.38487694e+01, 1.28603466e+01, 1.26333680e+01,
        9.53890114e+00, 4.42034417e+00, 3.42645780e+00, 2.40398565e+00,
        2.16695893e+00, 2.03597016e+00, 1.88330381e+00, 1.73351570e+00,
        1.67010315e+00, 1.63017694e+00, 1.57259164e+00, 1.50841239e+00,
        5.57176709e+00, 2.56347335e+00, 1.97252192e+00, 1.44516033e+00,
        1.32396112e+00, 1.27019282e+00, 1.19287951e+00, 1.13268097e+00,
        1.08903688e+00, 1.06914656e+00, 1.04211379e+00, 1.00482159e+00,
        9.05459197e+04, 2.44846297e+02, 9.10885257e+01)
    '''
    means = [0 for _ in range(39)]
    stds = [1 for _ in range(39)]
    '''
    
    trainset = Loader(root=data_path, flist=list_train, means=means, stds=stds)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=8)

    valset = Loader(root=data_path, flist=list_val, means=means, stds=stds)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=8)


    # Calculate loss
    criterion = nn.CrossEntropyLoss()

    optimizer = optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-15, epsilon=1e-5, verbose=1, mode='min')


    file_inf_train = os.path.join(args.model_path[0], 'inform_' + args.model_name[0][:-2] + 'train.log')
    file_inf_val = os.path.join(args.model_path[0], 'inform_' + args.model_name[0][:-2] + 'val.log')
    pd.DataFrame(columns=[
        'epoch', 
        'num_batch', 
        'loss', 
        'accuracy',  
        'agg_loss', 
        'agg_acc',  
        'lr', 
        'time_on_batch']).to_csv(file_inf_train, index=False)
    pd.DataFrame(columns=[
        'epoch', 
        'num_batch', 
        'loss', 
        'accuracy',  
        'agg_loss', 
        'agg_acc',  
        'lr', 
        'time_on_batch']).to_csv(file_inf_val, index=False)

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume[0]):
            print("=> loading checkpoint '{}'".format(args.resume[0]))
            checkpoint = torch.load(args.resume[0])
            start_epoch = checkpoint['epoch']
            model_dict = model.state_dict()
            predicted_dict = checkpoint['state_dict']
            predicted_dict = {k: predicted_dict[k] for k in model_dict if k in predicted_dict and k.find('fc') == -1}
            model_dict.update(predicted_dict)
            model.load_state_dict(model_dict)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume[0], checkpoint['epoch']))
                for g in optimizer.param_groups:
                    g['lr'] = lr
                print("Learning rate: ", lr)
            except: 
                print("Can't load optimizer state dict")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume[0]))

    number_epochs += start_epoch
    for_output = int(len(trainloader) // 10)
    len_train = len(trainloader)
    len_val = len(valloader)
    best_accuracy = 0.0
    best_loss = 1e10

    logger = Logger(logger_dir)
    step_train = 0
    step_val = 0

    # main cycle
    for epoch in range(start_epoch, number_epochs):
        start_t = dtm.now()
        print("Epoch: ", epoch + 1, "/", number_epochs)
        loss_acc_mat = np.zeros((len(trainloader), 8))
        running_loss = 0.0
        running_acc = 0.0
        total = 0.0
        # training
        model.train()
        for i, data in enumerate(trainloader, 0):
            start_batch = dtm.now()
            # get the inputs
            inputs, labels = data
            #print(inputs)
            #exit(0)
            
            # wrap them in Variable
            inputs, labels = Variable(inputs, volatile=False).cuda(), Variable(labels, volatile=False).cuda()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            total += labels.data.size(0)
            running_acc += (preds == labels.data).sum()
            delta = dtm.now() - start_batch
            print('     Epoch: %d / %d, batch: [ %d / %d ] loss: %.3f, accuracy: %2.2f, lr: %f, time: %f' %
                    ( epoch + 1, number_epochs, i + 1, len_train, running_loss / (i + 1), running_acc / total * 100.0, lr, delta.total_seconds() ))
            loss_acc_mat[i] = np.array([epoch + 1, i + 1, loss.data[0], (preds == labels.data).sum() / labels.data.size(0) * 100.0, running_loss / (i + 1),  running_acc / total * 100.0, lr, delta.total_seconds()])
            if i % 100 == 99:
                sys.stdout.flush()
                logging(loss.data[0], running_acc, total, is_train=True, step=step_train, is_per_epoch=False, inputs=inputs)
            step_train += 1

        delta = dtm.now() - start_t
        print("Total time: ", delta.total_seconds(), " s")
        
        pd.DataFrame(loss_acc_mat).to_csv(file_inf_train, mode='a', header=False, index=False)

        logging(loss.data[0], running_acc, total, is_train=True, step=epoch + 1, is_per_epoch=True)

        # validation
        loss_acc_mat = np.zeros((len(valloader), 8))
        running_acc = 0.0
        running_loss = 0.0
        total = 0.0
        print('Validation: ')
        model.eval()
        for i, data in enumerate(valloader, 0):
            start_batch = dtm.now()
            images, labels = data
            images, labels = Variable(images, volatile=True).cuda(), Variable(labels, volatile=True).cuda()
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.data[0]
            total += labels.data.size(0)
            running_acc += (preds == labels.data).sum()
            delta = dtm.now() - start_batch
            print('     Epoch: %d / %d, batch: [ %d / %d ] loss: %.3f, accuracy: %2.2f, lr: %f, time: %f' %
                    ( epoch + 1, number_epochs, i + 1, len_val, running_loss / (i + 1), running_acc / total * 100.0, lr, delta.total_seconds() ))
            loss_acc_mat[i] = np.array([epoch + 1, i + 1, loss.data[0], (preds == labels.data).sum() / labels.data.size(0) * 100.0, running_loss / (i + 1),  running_acc / total * 100.0, lr, delta.total_seconds()])
            if i % 100 == 99:
                sys.stdout.flush()
                logging(loss.data[0], running_acc, total, is_train=False, step=step_val, is_per_epoch=False, inputs=images)
            step_val += 1
        
        pd.DataFrame(loss_acc_mat).to_csv(file_inf_val, mode='a', header=False, index=False)

        print('Validation loss: %.3f, validation accuracy: %2.2f' % (running_loss / len(valloader), 100.0 * running_acc / total))
        logging(loss.data[0], running_acc, total, is_train=False, step=epoch + 1, is_per_epoch=True)
        
        cur_accuracy = 100.0 * running_acc / total
        cur_loss = running_loss / len(valloader)
        # saving
        if args.checkpoint:
            is_best = False
            if cur_loss < best_loss:
                best_loss = cur_loss
                is_best = True
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_loss' : best_loss
            }
            save_checkpoint(state, checkpoint_filename, is_best)
        scheduler.step(cur_loss, epoch + 1)
        lr = scheduler.optimizer.param_groups[0]['lr']

    print('Finished Training')
    print('Saving model...')
    torch.save(model, out_filename)
    print('Model was saved')
