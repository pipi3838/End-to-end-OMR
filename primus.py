import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import os
import numpy as np
from PIL import Image

class Init_dataset():
    def __init__(self, corpus_filepath, dictionary_path, val_split):
        #trainning + validation set id
        train_filepath = os.path.join(corpus_filepath, 'train.txt')
        corpus_file = open(train_filepath,'r')
        train_list = corpus_file.read().splitlines()
        corpus_file.close()
        
        #test set id
        test_filepath = os.path.join(corpus_filepath, 'test.txt')
        corpus_file = open(test_filepath,'r')
        self.test_list = corpus_file.read().splitlines()
        corpus_file.close()        

        # Dictionary: 字典的semantic word 轉 int; int 轉 sementic word
        self.word2int = {}
        self.int2word = {}

        dict_file = open(dictionary_path,'r')
        dict_list = dict_file.read().splitlines()
        for word in dict_list:
            if not word in self.word2int:
                word_idx = len(self.word2int) + 1  #0:留給空白
                self.word2int[word] = word_idx
                self.int2word[word_idx] = word
        dict_file.close()

        # Train and validation split
        random.shuffle(train_list) 
        val_idx = int(len(train_list) * val_split) 
        self.training_list = train_list[val_idx:]
        self.validation_list = train_list[:val_idx]

        # self.training_list = train_list[:50]
        # self.validation_list = train_list[50:60]

        print('Semantic Vocabulary with ' + str(len(self.word2int))) 
        
class CTC_PriMuS(Dataset):   
    def __init__(self, corpus_path, semantic, distortions, mode, set_list, w2i, i2w, img_height):
        self.semantic = semantic
        self.distortions = distortions
        self.corpus_path = corpus_path        
        self.mode = mode
        self.set_list = set_list
        self.word2int = w2i
        self.int2word = i2w
        self.img_height = img_height
        self.len_words = len(w2i)  #1781個字

    def __len__(self):
        return len(self.set_list)
    
    def __getitem__(self,idx):
        img_name = self.set_list[idx]
        img_path = os.path.join(self.corpus_path, img_name, img_name)
        
        if self.distortions:  #有distortion版本
            img = Image.open(img_path + '_distorted.jpg').convert('L')
        else:
            img = Image.open(img_path + '.png').convert('L')
        if img is None:
            raise Exception('{} file not found'.format(img_path))
            
        width = int(float(self.img_height * img.size[0]) / img.size[1])
        img = transforms.Resize((self.img_height,width))(img)
        img = transforms.ToTensor()(img)
        img = (1. - img)/1.
            
            
        #Ground Truth 
        if self.semantic: #做semantic encoding
            label_path = img_path + '.semantic'
        else:  #做agnostic encoding
            label_path = img_path + '.agnostic'
        label_file = open(label_path, 'r')
        label_plain = label_file.readline().rstrip().split('\t')
        label_file.close()  
        label = [self.word2int[lab] for lab in label_plain]
        
        true_len = len(label)  #sequence長度
        
        return img, label, true_len
        
def collate_fn(batch):
    # Transform to batch
    imgs = [item[0] for item in batch]  #樂譜batch
    img_widths = [img.shape[2] for img in imgs]  #取出該batch中，所有樂譜的寬度
    max_image_width = max(img_widths)  #取出batch中最寬的照片
    batch_img = np.zeros((len(imgs), imgs[0].shape[0], imgs[0].shape[1], max_image_width),dtype=np.float32)  #[bs, C, H, W]
    for i, img in enumerate(imgs):
        batch_img[i, 0, :, 0:img.shape[2]] = img
    batch_img = torch.tensor(batch_img, dtype=torch.float32)

    sequences = [item[1] for item in batch]  #semantic encoding
    sequence_width = [len(seq) for seq in sequences]
    max_sequence_width = max(sequence_width)
    target = torch.zeros((len(sequences),max_sequence_width),dtype=torch.int32)   
    for i,seq in enumerate(sequences):
        target[i, :len(seq)] = torch.tensor(seq, dtype=torch.int32) 
    
    true_len = [item[2] for item in batch]   #紀錄每個data的真實sequence長度
    true_len = torch.tensor(true_len, dtype=torch.int32)
    
    
    return [batch_img, target, true_len]
         