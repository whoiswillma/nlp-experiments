import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import allennlp.modules.conditional_random_field as crf
from typing import List, Tuple, Optional, Dict

class BiLSTM_CRF(nn.Module):
    def __init__(self, device:torch.device, vocab_size:int, num_tags:int,
                 embedding_dim:int, lstm_hidden_dim:int, lstm_num_layers:int,
                 dropout:float,constraints:Optional[List[Tuple[int, int]]],
                 pad_idx:int):
        super(BiLSTM_CRF, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.constraints=constraints
        self.pad_idx = pad_idx
        # TODO: change to pretrained embeddings
        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embedding_dim
        )
        # LSTM + Linear Layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
            num_layers=lstm_num_layers
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(
            in_features=lstm_hidden_dim,
            out_features=self.num_tags
        )
        # Conditional Random Field for Decoding
        self.crf = crf.ConditionalRandomField(
            num_tags=self.num_tags,
            constraints=self.constraints
        )

    def create_mask(self, src:torch.LongTensor) -> torch.LongTensor:
        mask = (src != self.pad_idx).permute(0, 1)
        return mask

    def forward(self, input:torch.LongTensor, input_lens:torch.LongTensor,
                labels:torch.LongTensor, decode:bool) -> Dict[str, any]:
        # pass through bilstm
        embedded = self.dropout(self.embeddings(input))
        packed_embedded = rnn.pack_padded_sequence(embedded, input_lens, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.lstm(packed_embedded)
        output, _ = rnn.pad_packed_sequence(packed_output, batch_first=True)
        self.output = self.dropout(output)
        output = self.linear(output)
        # pass through crf
        mask = self.create_mask(input)
        result = {}
        if decode:
            result['tags'] = self.crf.viterbi_tags(logits=output, mask=mask)
            print(result['tags'])
        else:
            result['loss'] = -self.crf(inputs=output, tags=labels, mask=mask)
        return result