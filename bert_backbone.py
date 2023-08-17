import torch
import numpy as np
from torch import nn

from transformers import BertModel, BertTokenizer

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.init_ffnn(self.latent_dim)

    def init_ffnn(self, latent_dim, inp_dim=1024, hid_dims=None):
        # initialize perceptron or MLP
        if hid_dims is None:
            self.model = nn.Sequential(
                nn.Linear(inp_dim, latent_dim)
            )
        elif len(hid_dims) == 1:
            self.model = nn.Sequential(
                nn.Linear(inp_dim, hid_dims[0]),
                nn.ReLU(),
                nn.Linear(hid_dims[0], latent_dim)
            )
        elif len(hid_dims) == 2:
            self.model = nn.Sequential(
                nn.Linear(inp_dim, hid_dims[0]),
                nn.ReLU(),
                nn.Linear(hid_dims[0], hid_dims[1]),
                nn.ReLU(),
                nn.Linear(hid_dims[1], latent_dim)
            )
        elif len(hid_dims) == 3:
            self.model = nn.Sequential(
                nn.Linear(inp_dim, hid_dims[0]),
                nn.ReLU(),
                nn.Linear(hid_dims[0], hid_dims[1]),
                nn.ReLU(),
                nn.Linear(hid_dims[1], hid_dims[2]),
                nn.ReLU(),
                nn.Linear(hid_dims[2], latent_dim)
            )
        else:
            print("The model can have up to three layers. The remaining layers will be ignored.")
            self.model = nn.Sequential(
                nn.Linear(inp_dim, hid_dims[0]),
                nn.ReLU(),
                nn.Linear(hid_dims[0], hid_dims[1]),
                nn.ReLU(),
                nn.Linear(hid_dims[1], hid_dims[2]),
                nn.ReLU(),
                nn.Linear(hid_dims[2], latent_dim),
                nn.Sigmoid()
            )
        self.model.apply(self.init_weights)

    def forward(self, x):
        x = self.model(x)
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

class HfBertModel(nn.Module):
    def __init__(self, device='cuda:0', CLS_representation=True, multiple_layers=False,
                 weights_name="bert-large-uncased"):
        super().__init__()
        self.max_input_len = 1024
        self.weights_name = weights_name
        self.bert = BertModel.from_pretrained(self.weights_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.weights_name)
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.CLS_representation = CLS_representation
        self.multiple_layers = multiple_layers
        self.device = device

    def concatenate_layers(self, reps):
        pooled_output = torch.cat(tuple([reps.hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)[:, 0, :]
        return pooled_output  # 4 * hidden_dim, because we concat 4 layers

    def forward(self, indices, mask):
        reps = self.bert(indices, attention_mask=mask)
        latent_vector = self.concatenate_layers(reps) if self.multiple_layers else reps.last_hidden_state.squeeze(0)
        representation = reps.pooler_output if self.CLS_representation else torch.mean(latent_vector, dim=0).unsqueeze(0)
        return representation

    def batch_score(self, tags_batch):
        tokenization = self.tokenizer(
            [". ".join(tags) + "." for tags in tags_batch],
            padding="longest",
            truncation="longest_first",
            max_length=self.max_input_len,
            return_tensors="pt",
        )
        input_ids = tokenization.input_ids.to(self.device)
        attention_mask = tokenization.attention_mask.to(self.device)
        representations = self(input_ids, attention_mask)
        return representations  # type: ignore

    def batch_score_split(self, tags_batch):
        tokenization = self.tokenizer(
            tags_batch,
            padding="longest",
            truncation="longest_first",
            max_length=self.max_input_len,
            return_tensors="pt",
        )
        input_ids = tokenization.input_ids.to(self.device)
        attention_mask = tokenization.attention_mask.to(self.device)
        representations = self(input_ids, attention_mask)
        return representations  # type: ignore


class HfBertQueryModel(HfBertModel):
    def __init__(self, device='cuda:0', CLS_representation=True, multiple_layers=False,
                 weights_name="bert-large-uncased"):
        super().__init__(device=device, CLS_representation=CLS_representation, multiple_layers=multiple_layers,
                         weights_name=weights_name)

    def forward(self, indices, mask):
        reps = self.bert(indices, attention_mask=mask)
        latent_vector = self.concatenate_layers(reps) if self.multiple_layers else reps.last_hidden_state.squeeze(0)
        representation = reps.pooler_output if self.CLS_representation else torch.mean(latent_vector, dim=1)
        return representation
