import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from model_baseline import DNN
from getter_dataset import MfccLoader
#from model_32_conv_merge_bn_softmax_py import  model_32_conv_merge_bn_softmax_py, Lambda
#from mbv_good_bad import model_25softmax_py

# parsing
parser = argparse.ArgumentParser(description='Predict by model.')
parser.add_argument('--model_seq',
                    required=True,
                    help='file with model\'s weights')
                
parser.add_argument('--checkpoint',
                    nargs=1,
                    help='file with model\'s checkpoint')
                
parser.add_argument('--batch_size',
                    nargs=1,
                    default=[1],
                    type=int,
                    help='size of batch')

parser.add_argument('--list',
                    nargs=1,
                    required=True,
                    help='image list filename')

parser.add_argument('--list_path',
                    nargs=1,
                    required=True,
                    help='path to file with image list')

parser.add_argument('--out_info_path',
                    nargs=1,
                    required=True,
                    help='path to results file')

args = parser.parse_args()



# getting image list filename
list_filename = args.list[0]

# getting image list path
list_path = args.list_path[0]

# getting train/val filenames
list_fullpath = os.path.join(list_path, list_filename)
num_classes = pd.read_csv(list_fullpath)
num_classes = num_classes['lbl'].max()

model_seq = args.model_seq.split(',')
model_seq = [int(i) for i in model_seq]
model = DNN(model_seq)
model.cuda()

# getting checkpoint
if args.checkpoint:
    checkpoint = torch.load(args.checkpoint[0])
    epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])

# getting batch size
batch_size = args.batch_size[0]

# getting out info path
out_info_file = args.out_info_path[0]

print('\n')
print("Model: ", args.model_seq)
if args.checkpoint:
    print("Checkpoint: ", args.checkpoint)
print("Image list file: ", list_fullpath)
print("Size of batch: ", batch_size)
print("Results file path: ", out_info_file)
print("Number of epoch: ", epoch)
print("Best loss: ", best_loss)
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

testset = MfccLoader(flist=list_fullpath, means=means, stds=stds)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=8)

correct = 0.
total = 0.

scrs = [ str(i) + '_scr' for i in range(1, num_classes + 1)]
lbls = [ str(i) + '_lbl' for i in range(1, num_classes + 1)]

df_out_test = pd.DataFrame(columns=['true_lbl'] + scrs + lbls)

model.eval()
for i, data in enumerate(testloader, 0):
    images, labels = data
    images, labels = Variable(images, volatile=True).cuda(), Variable(labels, volatile=True).cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.data.size(0)
    correct += (predicted == labels.data).sum()

    outputs = outputs.data#F.softmax(outputs, 1).data
    _, topk_lbl = torch.topk(outputs, outputs.size()[1])
    sort_out = np.sort(outputs.cpu().numpy(), axis=1)[:, ::-1]
    df = pd.concat([pd.DataFrame(labels.data.cpu().numpy() + 1, columns=['true_lbl']), pd.DataFrame(sort_out, columns=scrs), pd.DataFrame(topk_lbl.cpu().numpy() + 1, columns=lbls)], axis=1)
    df_out_test = pd.concat([df_out_test, df], ignore_index=True)
    if i % 100 == 99:
        print('Progress: %d / %d' % (i+1, len(testloader)), end='\r')

print('Accuracy on test images: %.4f %%' % (100. * correct / total))

print("Saving result...")
df_out_test.to_csv(out_info_file, index=None)
print("Results was saved")
