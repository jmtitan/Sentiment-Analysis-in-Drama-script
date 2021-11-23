import torch
from train import EmotionClassifier, RoleDataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel
PRE_TRAINED_MODEL_NAME = './pretrained/bert-base-chinese'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RoletestDataset(Dataset):
    def __init__(self,ids, texts, tokenizer, max_len):
        self.texts = texts
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        text = str(self.texts[item])
        id = str(self.ids[item])
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
            'ids': id,
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        return sample



if __name__ == '__main__':
    model = torch.load('./ccf2.pth')

    test = pd.read_csv('data/test_dataset.tsv', sep='\t',encoding='gbk')
 
    mydataset = RoletestDataset(
                            ids= test['id'],
                            texts=test['content'],
                            max_len=512,
                            tokenizer=BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME))

    data_loader = DataLoader(dataset=mydataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=2)
    result = dict()
    for sample in tqdm(data_loader):
        input_ids = sample["input_ids"].to(device)
        attention_mask = sample["attention_mask"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        id = sample["ids"][0]
        love = outputs['love'].argmax(1).item()
        joy = outputs['joy'].argmax(1).item()
        fright = outputs['fright'].argmax(1).item()
        anger = outputs['anger'].argmax(1).item()
        fear = outputs['fear'].argmax(1).item()
        sorrow = outputs['sorrow'].argmax(1).item()
        tmp = str(love)+','+str(joy)+','+str(fright)+','+str(anger)+','+str(fear)+','+str(sorrow)
        result[id] = tmp
    df = pd.DataFrame.from_dict(result, orient='index')
    df.to_csv('./result.csv')
    
