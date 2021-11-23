import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np

from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel
PRE_TRAINED_MODEL_NAME = './pretrained/bert-base-chinese'

EPOCHS = 1  # 训练轮数
num_classes = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class RoleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        target_cols = ['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']
        text = str(self.texts[item])
        label = self.labels.loc[item,target_cols]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        # print(encoding['input_ids'])
        sample = {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        for label_col in target_cols:
            sample[label_col] = torch.tensor(label[label_col], dtype=torch.float)
        return sample
    
class EmotionClassifier(nn.Module):
    def __init__(self, n_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.out_love = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_joy = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_fright = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_anger = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_fear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_sorrow = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(1)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict = False
        )
        love = self.softmax(self.out_love(pooled_output))
        joy = self.softmax(self.out_joy(pooled_output))
        fright = self.softmax(self.out_fright(pooled_output))
        anger = self.softmax(self.out_anger(pooled_output))
        fear = self.softmax(self.out_fear(pooled_output))
        sorrow = self.softmax(self.out_sorrow(pooled_output))
        return {
            'love': love, 'joy': joy, 'fright': fright,
            'anger': anger, 'fear': fear, 'sorrow': sorrow,
        }
    
def data_train():
    with open('data/train_dataset_v2.tsv', 'r', encoding='gbk') as handler:
        lines = handler.read().split('\n')[1:-1]
        data = list()
        for line in tqdm(lines):
            sp = line.split('\t')
            if len(sp) != 4:
                print("ERROR:", sp)
                continue
            data.append(sp)
    train = pd.DataFrame(data)
    train.columns = ['id', 'content', 'character', 'emotions']

    test = pd.read_csv('data/test_dataset.tsv', sep='\t',encoding='gbk')
    submit = pd.read_csv('data/submit_example.tsv', sep='\t')
    train = train[train['emotions'] != '']

    train['emotions'] = train['emotions'].apply(lambda x: [int(_i) for _i in x.split('\"')[1].split(',')])
    train[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = train['emotions'].values.tolist()
    train.index = [i for i in range(train.index.size)]
    
    mydataset = RoleDataset(texts=train['content'],
                            labels=train[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']],
                            max_len=512,
                            tokenizer=BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME))
    return mydataset

def num2eye(n, num_classes):
    eye = np.zeros([len(n),num_classes])
    for i in range(len(n)):
        eye[i,int(n[i])] = 1
    return torch.tensor(eye).to(torch.float32)

if __name__ == '__main__':
    mydataset = data_train()
    model = EmotionClassifier(num_classes).to(device)
    train_data_loader = DataLoader(dataset=mydataset,
                                    batch_size=2,
                                    shuffle=False,
                                    num_workers=2)

    optimizer = AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    criterion = nn.MSELoss().to(device)

    model = model.train()
    losses = []
    correct_predictions = 0
    for sample in tqdm(train_data_loader):
        input_ids = sample["input_ids"].to(device)
        attention_mask = sample["attention_mask"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss_love = criterion(outputs['love'], num2eye(sample['love'].numpy(), num_classes).to(device))
        loss_joy = criterion(outputs['joy'], num2eye(sample['joy'].numpy(), num_classes).to(device))
        loss_fright = criterion(outputs['fright'], num2eye(sample['fright'].numpy(), num_classes).to(device))
        loss_anger = criterion(outputs['anger'], num2eye(sample['anger'].numpy(), num_classes).to(device))
        loss_fear = criterion(outputs['fear'], num2eye(sample['fear'].numpy(), num_classes).to(device))
        loss_sorrow = criterion(outputs['sorrow'], num2eye(sample['sorrow'].numpy(), num_classes).to(device))
        loss = loss_love + loss_joy + loss_fright + loss_anger + loss_fear + loss_sorrow
    #     print(loss)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print(f'MSE:{np.mean(losses)}')
    #MSE:1.4532506393163935
    #MSE:0.23166099007008584
    torch.save(model, "ccf2.pth") 
