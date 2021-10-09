from collections import defaultdict
from typing import Optional, List, Tuple, Union

import allennlp.modules.conditional_random_field as crf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from datasets import Dataset


class BiLstmCrfModel(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            num_tags: int,
            embedding_dim: int = 300,
            embeddings: Optional[torch.Tensor] = None,
            freeze_embeddings: Optional[bool] = False,
            lstm_hidden_dim: int = 300,
            crf_constraints: Optional[List[Tuple[int, int]]] = None
    ):
        super(BiLstmCrfModel, self).__init__()

        assert lstm_hidden_dim % 2 == 0

        if embeddings != None:
            assert embeddings.shape[1] == embedding_dim
            self._embedding = nn.Embedding.from_pretrained(
                embeddings=embeddings,
                freeze=freeze_embeddings
            )
        else:
            assert not freeze_embeddings
            self._embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim
            )


        self._lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )

        self._linear = nn.Linear(
            in_features=lstm_hidden_dim,
            out_features=num_tags
        )

        self._crf = crf.ConditionalRandomField(
            num_tags=num_tags,
            constraints=crf_constraints
        )

    def forward(
            self,
            token_ids: torch.Tensor,
            seq_lens: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            decode_tags: bool = False
    ) -> dict[str, any]:
        assert labels != None or decode_tags

        total_length = token_ids.shape[1]
        embeddings = self._embedding(token_ids)

        packed_input = rnn.pack_padded_sequence(
            embeddings,
            seq_lens,
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, _ = self._lstm(packed_input)

        output, output_lens = rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=total_length
        )

        output = self._linear(output)

        output = F.softmax(output, output.dim() - 1)

        result_dict = {}
        max_len = seq_lens.max().item()
        mask = torch.arange(max_len).expand(len(seq_lens), max_len) < seq_lens.unsqueeze(1)

        if labels != None:
            result_dict['loss'] = -self._crf(
                inputs=output,
                tags=labels,
                mask=mask
            )

        if decode_tags:
            result_dict['tags'] = self._crf.viterbi_tags(
                logits=output,
                mask=mask
            )

        return result_dict


def generate_token_to_idx_dict(dataset) -> defaultdict[str, int]:
    """
    Unknown words are mapped to the zero index.
    """

    vocab = set()
    for example in dataset:
        for token in example['tokens']:
            vocab.add(token)

    result = defaultdict(int)
    for index, token in enumerate(sorted(vocab)):
        result[token] = index + 1

    return result


def make_inputs(
        examples: Union[list, Dataset],
        token_to_idx: dict[str, int],
        compute_loss: bool = False,
        decode_tags: bool = False
) -> dict[str, any]:

    def map_tokens_to_tensor(tokens: list[str]) -> torch.LongTensor:
        return torch.LongTensor([token_to_idx[token] for token in tokens])

    result = {
        'token_ids': rnn.pad_sequence(
            [map_tokens_to_tensor(example['tokens']) for example in examples],
            batch_first=True,
            padding_value=0
        ),
        'seq_lens': torch.tensor([len(example['tokens']) for example in examples])
    }

    if compute_loss:
        result['labels'] = rnn.pad_sequence(
            [torch.tensor(example['ner_tags']) for example in examples],
            batch_first=True,
            padding_value=0
        )

    result['decode_tags'] = decode_tags

    return result


def make_stats():
    return {
        'loss': 0.0,
        'num_examples': 0
    }


def backprop(
        model,
        examples: Union[list, Dataset],
        token_to_idx: dict[str, int],
        stats: dict[str, any]
):
    model.train()

    inputs = make_inputs(examples, token_to_idx, compute_loss=True)
    output = model(**inputs)
    loss = output['loss']
    loss.backward()

    stats['loss'] += loss.item()
    stats['num_examples'] += len(examples)
