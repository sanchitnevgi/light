import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import transformers
from transformers import AlbertTokenizer
import nlp

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class PolarNet(LightningModule):
    def __init__(self):
        super(PolarNet, self).__init__()
        
        self.embedding = nn.Embedding(30000, 300)

        self.lstm = nn.LSTM(300, 768, batch_first=True)

        self.linear1 = nn.Linear(768, 100)
        self.linear2 = nn.Linear(100, 2)
        
    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, c_n) = self.lstm(x)

        h_n = h_n.squeeze(dim=0)
        h_n = self.linear1(h_n)
        h_n = F.relu(h_n)
        h_n = self.linear2(h_n)
        h_n = F.relu(h_n)

        return h_n
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch['text'], batch['label']
        
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)

        tensorboard_logs = {'train_loss': loss}
        
        return {'loss': loss, 'log': tensorboard_logs}
        
    def train_dataloader(self):
        return DataLoader(train, batch_size=16, num_workers=4, shuffle=True)
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['text'], batch['label']
        
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)

        return {'val_loss': loss}
    
    def val_dataloader(self):
        return DataLoader(val, batch_size=16, num_workers=4)
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch['text'], batch['label']
        
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)
        
        return {'test_loss': loss}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}

        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}
    
    def test_loader(self):
        return DataLoader(test, batch_size=16, num_workers=4)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

def main():
    tb_logger = loggers.TensorBoardLogger('logs/')

    logger.info('Dowloading the Yelp Polarity dataset')
    yelp = nlp.load_dataset('yelp_polarity')
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    train, val, test = nlp.load_dataset('yelp_polarity', split=['train[:90%]', 'train[-10%:]', 'test'])

    def _tokenize(example):
        text = example['text']
        text = text.strip().replace('\\""', '').replace('\n', '')
        text = tokenizer.encode(text, max_length=256, pad_to_max_length=True)
        
        return { 'text': text }

    logger.info('Converting features for train/val/test')

    train = train.map(lambda example: _tokenize(example))
    train.set_format(type='torch')

    val = val.map(lambda example: _tokenize(example))
    val.set_format(type='torch')

    test = test.map(lambda example: _tokenize(example))
    test.set_format(type='torch')

    model = PolarNet()

    trainer = Trainer(gpus=1, logger=tb_logger)
    trainer.fit(model)
    trainer.test()

if __name__ == "__main__":
    main()