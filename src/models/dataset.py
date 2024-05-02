import torch
from torch.utils.data import Dataset


class RuBERTDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_seq_len = 512):
        self.texts = texts
        self.data_y = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __getitem__(self, index):
        text = self.texts[index]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        if self.data_y is not None:
            return {'ids': torch.tensor(ids),
                    'mask': torch.tensor(mask)}, torch.tensor(self.data_y.toarray()[index], dtype=float)
        else:
            return {
                'ids': torch.tensor(ids),
                'mask': torch.tensor(mask),
            }

    def __len__(self) -> int:
        return len(self.texts)
