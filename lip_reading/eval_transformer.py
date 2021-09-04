

from tqdm import tqdm
import torch
import numpy as np
import os

from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch import nn
from transformer import Transformer
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from nlp_metrics.eval import EvalCap
import fastwer

from Dataset import Dataset,PadCollate
from helpers import text_to_labels, labels_to_text


# 1. "Initial the parameters"


# parameter definition

batch_size = 64
root = '/vol/bitbucket/bh1511/GRID_AV/data'
tokenizer = Tokenizer.from_file("/vol/bitbucket/bh1511/data/dataset/grid.json")
vf_root = '/vol/bitbucket/bh1511/data'
speaker_list = os.listdir(root)
test = ["s1","s2","s20","s22"]
validation = ["s3","s4","s23","s24"]
training = list(set(speaker_list) - set(test) - set(validation))
device = torch.device('cuda')
image_or_mesh = "image"
level = "word"
pos_input = 80

if level == "word":
    pos_target = 7
    vocab_size = tokenizer.get_vocab_size()
else:
    pos_target = 34
    vocab_size = 34


# post processor 
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)


# initial dataset
dataset = Dataset(root,vf_root, level, image_or_mesh)


# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))


val_indices = list(range(2*1000,4*1000)) + list(range(21*1000, 23*1000))
test_indices = list(range(2*1000)) + list(range(19*1000, 21*1000))
train_indices = list(set(indices) - set(val_indices) - set(test_indices))

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler  = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size, 
                                           sampler=train_sampler, 
                                           collate_fn=PadCollate(True,tokenizer,level, 0),
                                           num_workers = 8,
                                           pin_memory=True,
                                           persistent_workers=True)

validation_loader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=batch_size,
                                                sampler=valid_sampler,
                                                collate_fn=PadCollate(False,tokenizer, level, 0),
                                                num_workers = 8,
                                                pin_memory=True,
                                                persistent_workers=True)

test_loader = torch.utils.data.DataLoader(dataset, 
                                          batch_size=batch_size,
                                          sampler=test_sampler,
                                          collate_fn=PadCollate(False,tokenizer, level, 0),
                                          num_workers = 8,
                                          pin_memory=True,
                                          persistent_workers=True)



# 2. "initial transformer model and define loss function"

num_layers = 8
d_model = 512
dff = 2048
num_heads = 8
dropout_rate = 0.1

if image_or_mesh == "image":
    vid_feat_size=512
else:
    vid_feat_size=468*3
    
#
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    video_feature_size=vid_feat_size,
    target_vocab_size=vocab_size,
    pos_input=pos_input,
    pos_target=pos_target,
    rate=dropout_rate,
    device=device
)


loss_object = torch.nn.CrossEntropyLoss(reduction='none')

# label smoothing
class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def loss(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) *             self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1)
        return loss
        
    def forward(self, input, target):
        # change to 30
        if level == "word":
            mask = torch.logical_not(torch.eq(target, 3))
        else:
            mask = torch.logical_not(torch.eq(target, 30))
        loss_ = self.loss(input, target)
        mask = mask.type(dtype=loss_.dtype)
        loss_ *= mask
        return torch.sum(loss_)/torch.sum(mask)
    


def loss_function(real, pred):
    # change to 30
    if level == "word":
        mask = torch.logical_not(torch.eq(real, 3))
    else:
        mask = torch.logical_not(torch.eq(real, 30))
    loss_ = loss_object(pred.permute([0, 2, 1]), real)
    mask = mask.type(dtype=loss_.dtype)
    loss_ *= mask
    return torch.sum(loss_)/torch.sum(mask)


def accuracy_function(real, pred, training):
    if level == "word":
        mask = torch.logical_not(torch.eq(real, 3))
    else:
        mask = torch.logical_not(torch.eq(real, 30))  
        
    if training == True:
        accuracies = torch.eq(real, torch.argmax(pred, dim=2))
    else:
        accuracies = torch.eq(real, pred)

    accuracies = torch.logical_and(mask, accuracies)
    accuracies = accuracies.type(dtype=torch.float32)
    mask = mask.type(dtype=torch.float32)
    acc = torch.sum(accuracies)/torch.sum(mask)
    return acc


optimizer = torch.optim.Adam(transformer.parameters(), lr=0.00001, betas=(0.9, 0.999))


class CustomSchedule(object):
    def __init__(self, _d_model, warmup_steps=1000):
        super(CustomSchedule, self).__init__()
        self.d_model = _d_model
        self.warmup_steps = warmup_steps

    def adjust_learning_rate(self, optim, step):
        arg1 = np.reciprocal(np.sqrt(step+3000))
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = np.reciprocal(np.sqrt(self.d_model)) * np.minimum(arg1, arg2)
        for param_group in optim.param_groups:
            param_group['lr'] = lr
        return lr

# scheduler = CustomSchedule(d_model, warmup_steps=1000)
label_sm =  LabelSmoothLoss(0.3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,2,3,4], gamma=0.1)
data_loaders = {"train": train_loader, "val": validation_loader}
data_lengths = {"train": len(train_loader), "val": len(validation_loader)}



# 3. "Define the evaluation function"
def evaluate(output, enc_mask, level):
    if level == "word":   
        # The first token to the transformer should be the start token
        max_length = 6
        target = torch.from_numpy(np.array([[1]],dtype = np.int32)).to(device=device, non_blocking=True)
    else:
        max_length = 32
        target = torch.from_numpy(np.array([[31.]],dtype = np.int32)).to(device=device, non_blocking=True)

    for i in range(max_length):

        if level == "word":   
            # The first token to the transformer should be the start token
            dec_target_padding_mask = torch.eq(target, 3).type(torch.float32)
        else:
            dec_target_padding_mask = torch.eq(target, 30).type(torch.float32)

        look_ahead_mask = (1 - torch.ones((target.size(1), target.size(1))).tril()).to(device=device, non_blocking=True)
        combined_mask = torch.maximum(dec_target_padding_mask[:, None, None, :], look_ahead_mask).to(device=device, non_blocking=True)


        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(output,
                                                     target,
                                                     False,
                                                     enc_mask,
                                                     combined_mask,
                                                     None)


        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = torch.argmax(predictions, dim=-1)
        

        # return the result if the predicted_id is equal to the end token
#         if level == "word":
#             if predicted_id == 2:
#                 break
#         else:
#             if predicted_id == 32:
#                 break

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        target = torch.cat([target, predicted_id], dim=-1)

    return torch.squeeze(target, dim=0)[1:], attention_weights


# 4. "Start training"

global_step = 1

# loop over the dataset multiple times
for epoch in range(1):  
    
    # report the location of the mistake
    torch.autograd.set_detect_anomaly(True)
    # for phase in ['val', 'train']:
    for phase in ['train', 'val']:
        loader = data_loaders[phase]
        length = data_lengths[phase]
        
        t = tqdm(enumerate(loader), total=length)
        total_loss = 0
        
        if phase == 'train':
            for (i, (inp, tar, enc_mask, combined_mask)) in t:

                output = inp.to(device=device, non_blocking=True)
                tar = tar.to(device=device, non_blocking=True)
                enc_mask = enc_mask.to(device=device, non_blocking=True)
                combined_mask = combined_mask.to(device=device, non_blocking=True)
                tar_inp = tar[:, :-1]
                tar_real = tar[:, 1:]
                

                
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()

                # adjust learning rate for training step
                # current_lr = scheduler.adjust_learning_rate(optimizer, global_step)
                # Forward pass to get output/logits
                predictions, attention_weights = transformer(output, tar_inp,
                                             True,
                                             enc_mask,
                                             combined_mask,
                                             None)
                
                
                
                
                
                # Calculate Sparse Categorical Loss
                # loss = loss_function(tar_real, predictions)
                loss = label_sm(predictions, tar_real)
                acc = accuracy_function(tar_real, predictions, True)
                

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                optimizer.step()

                global_step += 1

                # Updating step parameters
                total_loss += loss.detach().item()

                lr = scheduler.optimizer.param_groups[0]['lr']

                t.set_description(f'>> Phase: {phase} '
                                  f'>> Global Step {global_step}: '
                                  f'loss={loss.detach().item():.4f}, '
                                  f'acc={acc:.4f} '
                                  f'lr={lr:.7f}')

            print(f'Phase: {phase} >> Epoch {epoch+1}: avg_step_loss={total_loss / length:.5f}')
            
        else:
            with torch.no_grad():
                for (i, (inp, tar, enc_mask)) in t:

                    tar = tar.to(device=device, non_blocking=True)
                    output = inp.to(device=device, non_blocking=True)
                    enc_mask = enc_mask.to(device=device, non_blocking=True)
                    sentences = []
                    for v in range(output.size(0)):
                        sentence, _ = evaluate(output[v].unsqueeze(0), enc_mask[v].unsqueeze(0), level)
                        sentences.append(sentence)

                    predictions = torch.stack(sentences, dim=0)
                    
                    real = tar[:, 1:-1]
                    acc = accuracy_function(real, predictions, False)
                    t.set_description(f'>> Phase: {phase} '
                      f'acc={acc:.4f} ')
                    
                    model_name = "transformer" + "_" +phase 
                    torch.save(transformer, model_name+'.pt')
    scheduler.step()
print('Finished Training')




# 5. "Start testing"

transformer_test = torch.load('transformer_val.pt',map_location=torch.device('cuda'))
t = tqdm(enumerate(test_loader), total=len(test_loader))
real_list = []
pre_list = []
with torch.no_grad():
    for (i, (inp, tar, enc_mask)) in t:
        tar = tar.to(device=device, non_blocking=True)
        output = inp.to(device=device, non_blocking=True)
        enc_mask = enc_mask.to(device=device, non_blocking=True)
        sentences = []
        for v in range(output.size(0)):
            sentence, _ = evaluate(output[v].unsqueeze(0), enc_mask[v].unsqueeze(0), level)
            sentences.append(sentence)

        predictions = torch.stack(sentences, dim=0)
        real = tar[:, 1:-1]

        acc = accuracy_function(real, predictions, False)

        real_list.append(real)
        pre_list.append(predictions)
        t.set_description(f'acc={acc:.4f}')




# 6. "Report the results"

# decode token ids to sentence
ground_truth = []
hypothesis = []

for i in range(len(real)):
    for j in range(real[i].size()[0]):
        if level == "word":
            ground_truth.append(tokenizer.decode(real_list[i][j].tolist()))
            hypothesis.append(tokenizer.decode(pre_list[i][j].tolist()))
        else:
            index = torch.sum(real_list[i][j]  != 30)-1
            ground_truth.append(labels_to_text(real_list[i][j][:index]))
            hypothesis.append(labels_to_text(pre_list[i][j][:index]))
# calculate wer and cer
# wer
print(fastwer.score(hypothesis, ground_truth))
# cer
print(fastwer.score(hypothesis, ground_truth, char_level=True))
# get metric
my_metric = EvalCap(ground_truth,hypothesis)
my_metric.evaluate()







