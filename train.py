from os import path
import logging
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import transformers
from transformers import AlbertTokenizer
import nlp

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, loggers, seed_everything

logger = logging.getLogger(__name__)

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

class ReviewDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer):
        if split not in { 'train', 'val', 'test' }:
            raise ValueError(f'Dataset split must be of type [train, val, test]. Provided f{split}')
        self.tokenizer = tokenizer
        self.split = split
        self.data = []
        
        logger.info(f'Reading {split} data')

        data_file = path.join(data_dir, f'{split}.tsv')
        with open(data_file) as f:
            lines = f.readlines()
            for line in lines:
                review, label = line.split('\t')
                self.data.append((review, label))
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        
        input_ids = self.tokenizer.encode(text, max_length=256, pad_to_max_length=True, return_tensors='pt').squeeze(0)
        label = torch.LongTensor([ int(label) ])

        return (input_ids, label)

    def __len__(self):
        return len(self.data)

class PolarNet(LightningModule):
    def __init__(self, hparams):
        super(PolarNet, self).__init__()

        self.hparams = hparams
        
        self.embedding = nn.Embedding(30000, self.hparams.embedding_dim)

        self.lstm = nn.LSTM(self.hparams.embedding_dim, 768, batch_first=True)

        self.linear1 = nn.Linear(768, 100)
        self.linear2 = nn.Linear(100, 2)

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--embedding-dim', default=300, help="Word embedding dimention", type=int)

        return parser
        
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
        inputs, labels = batch
        labels = labels.view(-1)
        
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)

        tensorboard_logs = {'train_loss': loss}
        
        return {'loss': loss, 'log': tensorboard_logs}
        
    def train_dataloader(self):
        train = ReviewDataset(self.hparams.data_dir, split='train', tokenizer=tokenizer)

        return DataLoader(train, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.view(-1)
        
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)

        return {'val_loss': loss}
    
    def val_dataloader(self):
        val = ReviewDataset(self.hparams.data_dir, split='val', tokenizer=tokenizer)
        return DataLoader(val, batch_size=self.hparams.batch_size, num_workers=4)
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.view(-1)
        
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)
        
        return {'test_loss': loss}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}

        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}
    
    def test_dataloader(self):
        test = ReviewDataset(self.hparams.data_dir, split='test', tokenizer=tokenizer)
        return DataLoader(test, batch_size=self.hparams.batch_size, num_workers=4)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

def main(hparms):
    seed_everything(42)

    tb_logger = loggers.TensorBoardLogger(save_dir= 'logs')

    model = PolarNet(hparams)

    trainer = Trainer(
        gpus=hparams.gpus, num_nodes=hparams.num_nodes, distributed_backend=hparams.distributed_backend,
        precision=hparams.precision, early_stop_callback=hparams.early_stop,
        deterministic=hparams.deterministic, fast_dev_run=hparams.fast_dev_run, logger=tb_logger
    )
    trainer.fit(model)
    trainer.test()

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser = PolarNet.add_model_specific_arguments(parser)

    # Program arguments
    parser.add_argument('--data-dir', required=True, help="Data directory", type=str)
    parser.add_argument('--log-dir', default="./logs", help="Directory to save Tensorboard logs", type=str)

    # Training arguments
    parser.add_argument('--learning-rate', default=0.001, help="Learning rate", type=float)
    parser.add_argument('--batch-size', default=16, help="Batch size per GPU", type=int)
    # Arguments used by the Lightning Trainer
    parser.add_argument('--gpus', default=0, help="The number of GPUs used for training", type=int)
    parser.add_argument('--num_nodes', default=1, help="The number of nodes used for training", type=int)
    parser.add_argument('--distributed_backend', default='ddp', help="Distributed computing backend", type=str)
    parser.add_argument('--precision', default=32, help="FP16/32 training", type=int)
    parser.add_argument('--early-stop', default=True, action='store_true', help="Early stop if loss non-decreasing")
    parser.add_argument('--deterministic', default=True, action='store_true', help="For reproducibility")
    parser.add_argument('--fast-dev-run', default=False, action='store_true', help="Sanity check")

    hparams = parser.parse_args()
    hparams.batch_size = hparams.batch_size * max(hparams.gpus, 1)

    main(hparams)