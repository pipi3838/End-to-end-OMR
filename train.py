import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import copy
import random
import matplotlib.pyplot as plt
import matplotlib
import cv2
import math
import glob
import os
import sys
import shutil
import csv
import scipy.ndimage
import numpy as np
import warnings
from tqdm import tqdm
from tensorboardX import SummaryWriter 


from utils import GreedyDecoder, levenshtein
from model import CRNN
from primus import Init_dataset, CTC_PriMuS, collate_fn

os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings('ignore')

##### Hyper Parameter #####
corpus_path = "/nfs/nas-5.1/wbcheng/End-to-end_OMR/data/"
set_path = "./Data"
vocabulary_path = "./Data/vocabulary_semantic.txt"
model_save_path = "/nfs/nas-5.1/wbcheng/End-to-end_OMR/models/origin"
config_path = "/nfs/nas-5.1/wbcheng/End-to-end_OMR/config/origin"
semantic = True
distortions = False
img_height = 128
batch_size = 32
n_works = 8
nEpochs = 64000
weight_decay = 1e-8
seed = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lr = 2e-3

writer = SummaryWriter(config_path)

##### Setting Seed #####
torch.manual_seed(seed)
np.random.seed(seed)
if device == 'cuda':
	torch.cuda.manual_seed(seed)


##### Loading Dataset #####
get_list = Init_dataset(set_path, vocabulary_path, val_split = 0.1)
train_list, valid_list, test_list, word2int, int2word = get_list.training_list, get_list.validation_list, get_list.test_list, get_list.word2int, get_list.int2word
print ('Training with ' + str(len(train_list)) + ' and validating with ' + str(len(valid_list)) + ' and testing with ' + str(len(test_list)))
voc_len  = len(word2int)

trainset = CTC_PriMuS(corpus_path, semantic, distortions, mode='train', set_list=train_list, w2i=word2int, i2w=int2word, img_height=img_height)
trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=n_works)  #[(bs, 1, 128, w),torch.Size([bs, x])]
valset = CTC_PriMuS(corpus_path, semantic, distortions, mode='val', set_list=valid_list, w2i=word2int, i2w=int2word, img_height=img_height)
valset_loader = DataLoader(valset,batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=n_works)  #[(bs, 1, 128, w),torch.Size([bs, x])]
testset = CTC_PriMuS(corpus_path, semantic, distortions, mode='test', set_list=test_list, w2i=word2int, i2w=int2word, img_height=img_height)
testset_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=n_works)  #[(bs, 1, 128, w),torch.Size([bs, x])]
'''
train_bs = next(iter(trainset_loader))
img,encode,true_len = train_bs  
print(img.shape)
print(encode)
print(true_len)
plt.figure(figsize=(40,40))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(img,nrow=2,padding=1,normalize=True).numpy(),(1,2,0)), cmap='gray')
plt.show() 
'''

###### Training Config #####
decoder = GreedyDecoder()
model = CRNN().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CTCLoss()

min_loss = 1e8
#Training
for epoch in range(nEpochs):
    epoch_loss = 0
    model.train()
    with tqdm(total=len(trainset_loader), desc='[Train]') as bar:
        for idx,(img,targets,true_len) in enumerate(trainset_loader): 
            img, targets, true_len = img.to(device), targets.to(device), true_len.to(device)
            optimizer.zero_grad()
            output = model(img)  #[w, bs, 1782]
            torch.cuda.empty_cache()
            seq_len = torch.tensor([output.shape[0]] * output.shape[1])
            
            loss = criterion(log_probs=output, targets=targets, input_lengths=seq_len, target_lengths=true_len)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().item()
            # raise Exception(img.shape, true_len.shape)
            bar.set_postfix({'loss' : '{0:1.5f}'.format(epoch_loss / (idx + 1))})
            bar.update()
        # print(loss)
    epoch_loss /= len(trainset_loader)
    print ('Loss value at epoch ' + str(epoch) + ':' + str(epoch_loss))
    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.flush()

    if epoch%1 == 0:
        with torch.no_grad():
            valid_loss = 0
            model.eval()
            
            val_edit, val_len, dist_list, target_list = 0,0,[],[]
            for idx,(img,targets,true_len) in enumerate(valset_loader): 
                img, targets, true_len = img.to(device), targets.to(device), true_len.to(device)
                batch_size = int(true_len.shape[0])
                output = model(img)  #[w, bs, 1782]
                torch.cuda.empty_cache()
                seq_len = torch.tensor([output.shape[0]] * output.shape[1])
                loss = criterion(log_probs=output, targets=targets, input_lengths=seq_len, target_lengths=true_len) 
                valid_loss += loss.cpu().item() 
                output = output.detach().permute(1,0,2)  #[bs, seq_len, |vocs|+1(blank)]
                
                decoded, max_probs = decoder.decode(output, true_len) 
                for i in range(targets.shape[0]):
                    target = targets[i][:true_len[i]].cpu().numpy().tolist()
                    target = ' '.join(list(map(str, target)))
                    distance = levenshtein(target,decoded[i])
                    # target = decoder.convert_np_to_string(target)
                    val_edit += distance
                    val_len += true_len[i].cpu().item()
                    if idx == 0: 
                        dist_list.append(distance)
                        target_list.append(target)


            # writer.add_text('target', target_list[0])
            writer.add_text('Result', 'Decode: {} \n Target: {}'.format('  '.join(decoded[0]), target_list[0]), epoch)
                        # print('targets:',target_list[i])
                        # print('decoded:',decoded[i])
                        # print('un decoded:',max_probs[i].cpu().numpy())
                        # print('distance:',dist_list[i])
    if epoch%5 == 0:
        torch.save(model.state_dict(), os.path.join(model_save_path ,'model_{}.pth'.format(epoch)))
                
    valid_loss /= len(valset_loader)       
    if valid_loss < min_loss:
        print('Save model!')
        torch.save(model.state_dict(), os.path.join(model_save_path ,'model_best.pth'))
      
    print ('Loss value at validation ' + str(epoch) + ':' + str(valid_loss))
    sample_error_rate = 1. * val_edit / len(valset)
    symbol_error_rate = 100. * val_edit / val_len
    print ('[Epoch ' + str(epoch) + '] ' + str(sample_error_rate) + ' (' + str(symbol_error_rate) + ' SER) ' + ' from ' + str(len(valset)) + ' samples.')         
    writer.add_scalar('Test/Loss', valid_loss, epoch)
    writer.add_scalar('Test/Sample_error_rate', sample_error_rate, epoch)
    writer.add_scalar('Test/Symbol_error_rate', symbol_error_rate, epoch)
    writer.flush()

writer.close()