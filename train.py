import logging
from argparse import ArgumentParser

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

logger = logging.getLogger(__name__)

logger.info('Dowloading the Yelp Polarity dataset')
yelp = nlp.load_dataset('yelp_polarity')
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

train, val, test = nlp.load_dataset('yelp_polarity', split=['train[:90%]', 'train[-10%:]', 'test'])

def _tokenize(text):
    text = text.strip().replace('\\""', '').replace('\n', '')
    text = tokenizer.encode(text, max_length=256, pad_to_max_length=True)
    return text

def create_train_features(example):
    return { 'text': _tokenize(example['text']) }

def create_valid_features(example):
    return { 'text': _tokenize(example['text']) }

def create_test_features(example):
    return { 'text': _tokenize(example['text']) }

logger.info('Converting features for train/val/test')

train = train.map(create_train_features)
train.set_format(type='torch')

val = val.map(create_valid_features)
val.set_format(type='torch')

test = test.map(create_test_features)
test.set_format(type='torch')

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
    parser = ArgumentParser()

    # Training arguments
    parser.add_argument('--num-epochs', required=True, help="The number of epochs", type=int)
    parser.add_argument('--batch-size', required=True, help="Batch size", type=int)
    parser.add_argument('--learning-rate', required=True, help="Learning rate", type=float)
    parser.add_argument('--log-dir', default="./logs", help="Directory to save Tensorboard logs", type=str)

    args = parser.parse_args()

    tb_logger = loggers.TensorBoardLogger('logs/')

    model = PolarNet()

    trainer = Trainer(gpus=1, logger=tb_logger)
    trainer.fit(model)
    trainer.test()

if __name__ == "__main__":
    main()