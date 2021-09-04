import h5py
import torch
import numpy as np
import os

from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset

from torch import nn
from transformer import Transformer
from helpers import text_to_labels, labels_to_text

from aligns import Align


class Dataset(Dataset):  
    def __init__(self, root, vf_root, level, image_or_mesh = "mesh", absolute_max_string_len = 32):
        self.absolute_max_string_len = absolute_max_string_len
        self.current_speaker = None
        self.vf_root = vf_root
        self.level = level
        self.root = root
        if image_or_mesh == "mesh":
            self.use_image = False
        elif image_or_mesh == "image":
            self.use_image = True
        else:
            raise ValueError('An invalid value of image_or_mesh has been input.')
        
    def __len__(self):
        
        return (34-1)*1000

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        
        self.current_speaker = (index//1000) + 1
        index = index % 1000
        
        if self.current_speaker >= 21:
            self.current_speaker = "s" + str(self.current_speaker+1)
        else:
            self.current_speaker = "s" + str(self.current_speaker)
            
        speaker_root = os.path.join(self.root,self.current_speaker)   
        input_root = os.path.join(speaker_root,'video')
        input_root = os.path.join(input_root,os.listdir(input_root)[0])
        output_root = os.path.join(speaker_root,'align')
        name = os.path.basename(os.listdir(output_root)[index]).split(".")[0]
        
        if self.use_image == True: ### to tvc
            image_root = os.path.join(self.vf_root,self.current_speaker)
            image_root = os.path.join(image_root,"{}.hdf5".format(name))
            if os.path.isfile(image_root):     
                vid_h5 = h5py.File(image_root, "r", driver = "core")
                video_features = torch.from_numpy(np.array(vid_h5['video_features'])).type(torch.float32) # (framecount, 512)
                # prepare label
                sub_root = os.path.join(output_root,"{}.align".format(name)) 
                align = Align(self.absolute_max_string_len, text_to_labels)
                align.from_file(sub_root)
                if self.level == "word":
                    word = align.sentence
                else:
                    word = torch.from_numpy(align.padded_label).type(dtype=torch.int64)
                return video_features, word
            else:
                video_features = torch.rand(75, 512)
                if self.level == "word":
                    word = "[PAD] [PAD] [PAD] [PAD] [PAD] [PAD]"
                else:
                    word = torch.cat((torch.tensor([31]), torch.ones(self.absolute_max_string_len)*30, torch.tensor([32])), 0).type(dtype=torch.int64)
                    word = word.type(dtype=torch.int64)
                return video_features, word
            
        else: #### to transformer
            facemesh_root = os.path.join(input_root,"{}.hdf5".format(name))
            if os.path.isfile(facemesh_root):    
                vid_h5 = h5py.File(facemesh_root, "r", driver = "core")
                face_landmarks = torch.from_numpy(np.array(vid_h5['face_landmarks'])).type(torch.float32)
                # (framecount, 468 x 3)
                face_landmarks = face_landmarks.view(face_landmarks.size(0),-1)
                # prepare label
                sub_root = os.path.join(output_root,"{}.align".format(name)) 
                align = Align(self.absolute_max_string_len,text_to_labels)
                align.from_file(sub_root)
                if self.level == "word":
                    word = align.sentence
                else:
                    word = torch.from_numpy(align.padded_label).type(dtype=torch.int64)
                return face_landmarks, word
            else:
                face_landmarks = torch.rand(75,468*3)
                if self.level == "word":
                    word = "[PAD] [PAD] [PAD] [PAD] [PAD] [PAD]"
                else:
                    word = torch.cat((torch.tensor([31]), torch.ones(self.absolute_max_string_len)*30, torch.tensor([32])), 0).type(dtype=torch.int64)
                    word = word.type(dtype=torch.int64)
                return face_landmarks, word


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, training, token , level, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.tokenizer = token
        self.level = level
        self.training = training
        
    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        
        ### needs repair

        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        X_batch = map(lambda x: self.pad_tensor(x[0], pad=max_len, dim=self.dim), batch)
        y_batch = map(lambda x: x[1], batch)

        # stack all
        X = torch.stack([ t for t in X_batch ],dim = 0)
        enc_mask = torch.stack([ self.create_mask(X[i]) for i in range(X.size(0))],dim = 0)
     
        if self.training == True:
            
            if self.level == "word":   
                y = [ t for t in y_batch ]
                self.tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
                encoding = self.tokenizer.encode_batch(y)
                output = torch.stack([ torch.from_numpy(np.array(t.ids)) for t in encoding],dim = 0)
                #dec_target_padding_mask = torch.stack([torch.from_numpy(1 - np.array(t.attention_mask)) for t in encoding],dim = 0)
                dec_target_padding_mask = torch.eq(output[:,:-1], 3).type(torch.float32)
            else:
                output = torch.stack([ t for t in y_batch ],dim = 0)
                dec_target_padding_mask = torch.eq(output[:,:-1], 30).type(torch.float32)

            look_ahead_mask = 1 - torch.ones((output[:,:-1].size(1), output[:,:-1].size(1))).tril()
            combined_mask = torch.maximum(dec_target_padding_mask[:, None, None, :], look_ahead_mask)

            return X, output, enc_mask[:, None, None, :], combined_mask
        
        else:
            if self.level == "word":
                
                y = [ t for t in y_batch ]
                self.tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
                encoding = self.tokenizer.encode_batch(y)
                output = torch.stack([ torch.from_numpy(np.array(t.ids)) for t in encoding],dim = 0)
                return X, output, enc_mask[:, None, None, :]
            else:
                output = torch.stack([ t for t in y_batch ],dim = 0)
                return X, output, enc_mask[:, None, None, :]
                
                
    
    def pad_tensor(self,vec,pad,dim):
        """
        args:
            vec - tensor to pad
            pad - the size to pad to
            dim - dimension to pad

        return:
            a new tensor padded to 'pad' in dimension 'dim'
        """
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        pad_tensor = torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
        return pad_tensor
    
    def create_mask(self,x):
        len = x.size(0)
        mask = torch.zeros(len)
        for i in range(len):
            if x[i].any().type(torch.float32) == 0:
                mask[i] = 1
        return mask


    def __call__(self, batch):
        return self.pad_collate(batch)