import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu
from torchvision import models

from random import random
import lightning as L

from .utils import remove_specials

class Encoder(nn.Module):
    """
    Encoder module for the Seq2Seq model.

    Args:
        input_channels (int): Number of input channels.
        hidden_dim (int): Dimension of the hidden layer.
        dropout (float): Dropout probability.

    Attributes:
        model (torchvision.models.ResNet): Pretrained ResNet model.
    """

    def __init__(self, input_channels, hidden_dim, dropout=0.3):
        super(Encoder, self).__init__()

        self.model = models.resnet101(weights='IMAGENET1K_V2', progress=False)

        self.model.fc = nn.Linear(self.model.fc.in_features, hidden_dim)
        self.freeze()

    def forward(self, x):
        """
        Forward pass of the encoder module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.model(x)
        # x.shape: [batch_size, output_dim]
        return x

    def _req_grad(self, b):
        """
        Set the requires_grad attribute of the parameters.

        Args:
            b (bool): True to enable gradient computation, False otherwise.
        """
        for name, params in self.named_parameters():
            if name.find('fc') != -1:
                break
            params.requires_grad = b

    def freeze(self):
        """
        Freeze the encoder by disabling gradient computation for all parameters except the last layer.
        """
        self._req_grad(False)

    def unfreeze(self):
        """
        Unfreeze the encoder by enabling gradient computation for all parameters.
        """
        self._req_grad(True)


class Decoder(nn.Module):
    """
    Decoder module for the Seq2Seq model.

    Args:
        output_dim (int): Dimension of the output.
        embed_dim (int): Dimension of the embedding.
        hidden_dim (int): Dimension of the hidden layer.
        num_layers (int): Number of layers in the LSTM.
        dropout (float): Dropout probability.
        padding_idx (int): Index used for padding.

    Attributes:
        embedding (torch.nn.Embedding): Embedding layer.
        rnn (torch.nn.LSTM): LSTM layer.
        fc_out (torch.nn.Linear): Linear layer for output.
        dropout (torch.nn.Dropout): Dropout layer.
    """

    def __init__(
        self,
        output_dim,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout,
        padding_idx,
    ):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_features, captions):
        """
        Forward pass of the decoder module.

        Args:
            img_features (torch.Tensor): Image features.
            captions (torch.Tensor): Captions.

        Returns:
            torch.Tensor: Output tensor.
        """
        captions = captions[:, :-1]  # remove <eos>

        emb = self.dropout(self.embedding(captions))
        # x.shape: [batch_size, max_length-1, embed_dim]

        x = torch.cat((img_features.unsqueeze(1), emb), dim=1)
        # x.shape: [batch_size, max_length, embed_dim]

        outputs, _ = self.rnn(x)
        # outputs.shape: [batch_size, max_length, hidden_dim]
        x = self.fc_out(outputs)

        # output.shape: [batch_size, max_length, output_dim]
        return x

    def sample(self, img_feat, max_sent_size, states=None):
        """
        Generate captions for the given image features using greedy search.

        Args:
            img_feat (torch.Tensor): Image features.
            max_sent_size (int): Maximum sentence size.
            states (tuple): Initial hidden and cell states.

        Returns:
            torch.Tensor: Generated captions.
        """
        res = []
        dec_in = img_feat.unsqueeze(1)

        for i in range(max_sent_size):
            hidden, states = self.rnn(dec_in, states)
            outputs = self.fc_out(hidden.squeeze(1))
            # outputs.shape: [batch_size, output_dim]

            preds = outputs.argmax(-1)
            res.append(preds)

            dec_in = self.embedding(preds)
            dec_in = dec_in.unsqueeze(1)

        return torch.stack(res, dim=1)


class Seq2Seq(L.LightningModule):
    """
    Seq2Seq model for image captioning.

    Args:
        input_dim (int): Dimension of the input.
        embed_dim (int): Dimension of the embedding.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output.
        num_layers (int): Number of layers in the LSTM.
        vocab (dict): Vocabulary dictionary.
        dropout (float): Dropout probability.
        max_sent_size (int): Maximum sentence size.
        config (dict): Configuration dictionary.

    Attributes:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        all_validation (list): List to store validation results.
        LR (float): Learning rate.
        label_smoothing (float): Label smoothing factor.
        weight_decay (float): Weight decay factor.
        loss_fn (torch.nn.CrossEntropyLoss): Loss function.
    """

    def __init__(
        self,
        input_dim,
        embed_dim,
        hidden_dim,
        output_dim,
        num_layers,
        vocab,
        dropout,
        max_sent_size,
        config,
    ):
        super(Seq2Seq, self).__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(input_dim, embed_dim, dropout)
        self.decoder = Decoder(output_dim, embed_dim, hidden_dim, num_layers, dropout, vocab['<pad>'])

        self.max_sent_size = max_sent_size
        self.vocab = vocab
        self.all_validation = []

        self.LR = config['train']['learning_rate']
        self.label_smoothing = config['train'].get('label_smoothing', 0)
        self.weight_decay = config['train'].get('weight_decay', 0)

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=vocab['<pad>'],
            label_smoothing=self.label_smoothing,
        )

    def forward(self, img, captions):
        """
        Forward pass of the Seq2Seq model.

        Args:
            img (torch.Tensor): Input image.
            captions (torch.Tensor): Captions.

        Returns:
            torch.Tensor: Output tensor.
        """
        img_feat = self.encoder(img)
        outputs = self.decoder(img_feat, captions)

        return outputs

    def training_step(self, batch, batch_idx):
        """
        Training step of the Seq2Seq model.

        Args:
            batch (tuple): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        img, caption = batch

        img_feat = self.encoder(img)
        preds = self.decoder(img_feat, caption)

        output_dim = preds.size(-1)
        loss = self.loss_fn(
            preds.view(-1, output_dim),
            caption.reshape(-1),
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the Seq2Seq model.

        Args:
            batch (tuple): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        img, caption, all_captions = batch

        img_feat = self.encoder(img)
        preds = self.decoder(img_feat, caption)

        output_dim = preds.size(-1)
        loss = self.loss_fn(
            preds.view(-1, output_dim),
            caption.reshape(-1),
        )

        # predict the caption for these images using greed search
        self.all_validation.append((self.decoder.sample(img_feat, self.max_sent_size), all_captions))
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        """
        Callback function called at the end of each validation epoch.
        Calculates the BLEU score for the generated captions.
        """
        candidate_corpus = []
        references_corpus = []
        for preds, all_captions in self.all_validation:
            preds = preds.cpu().numpy()
            for i in range(len(preds)):
                pred_token = self.vocab.lookup_tokens(preds[i])
                pred_token = remove_specials(pred_token)
                candidate_corpus.append(pred_token)

                all_caption = [remove_specials(cap) for cap in all_captions[i]]
                references_corpus.append(all_caption)

        bleu = corpus_bleu(references_corpus, candidate_corpus)
        self.log("bleu_score@4", bleu)
        self.all_validation.clear()

    def configure_optimizers(self):
        """
        Configure the optimizer for training the model.

        Returns:
            torch.optim.Optimizer: Optimizer object.
        """
        if self.weight_decay > 0:
            return torch.optim.AdamW(self.parameters(), lr=self.LR, weight_decay=self.weight_decay)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.LR, weight_decay=self.weight_decay)
