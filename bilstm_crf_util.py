from collections import defaultdict
from typing import Optional, List, Tuple, Callable

import allennlp.modules.conditional_random_field as crf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn


class BiLstmCrfModel(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            num_tags: int,
            embedding_dim: int = 300,
            embeddings: Optional[torch.Tensor] = None,
            freeze_embeddings: Optional[bool] = False,
            lstm_hidden_dim: int = 300,
            lstm_num_layers: int = 1,
            dropout: float = 0.2,
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
            bidirectional=True,
            num_layers=lstm_num_layers,
            dropout=dropout
        )

        self._dropout = nn.Dropout(
            p=dropout
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

        output = self._dropout(output)

        output = self._linear(output)

        output = F.log_softmax(output, output.dim() - 1)

        result_dict = {}
        max_len = seq_lens.max().item()
        mask = torch.arange(max_len).expand(len(seq_lens), max_len) < seq_lens.unsqueeze(1)

        if labels != None:
            # crf.forward computes log likelihood, so we need to negate it
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


def generate_token_to_idx_dict(dataset: list[list[str]]) -> defaultdict[str, int]:
    """
    Unknown words are mapped to the zero index.
    """

    vocab = set()
    for tokens in dataset:
        for token in tokens:
            vocab.add(token)

    result = defaultdict(int)
    for index, token in enumerate(sorted(vocab)):
        result[token] = index + 1

    return result


def make_inputs(
        dataset_tokens: list[list[str]],
        dataset_ner_tags: list[list[int]],
        token_to_idx: dict[str, int],
        compute_loss: bool = False,
        decode_tags: bool = False
) -> dict[str, any]:

    assert len(dataset_tokens) == len(dataset_ner_tags)

    def map_tokens_to_tensor(tokens: list[str]) -> torch.LongTensor:
        return torch.LongTensor([token_to_idx[token] for token in tokens])

    result = {
        'token_ids': rnn.pad_sequence(
            [map_tokens_to_tensor(tokens) for tokens in dataset_tokens],
            batch_first=True,
            padding_value=0
        ),
        'seq_lens': torch.tensor([len(tokens) for tokens in dataset_tokens])
    }

    if compute_loss:
        result['labels'] = rnn.pad_sequence(
            [torch.tensor(ner_tags) for ner_tags in dataset_ner_tags],
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
        dataset_tokens: list[list[str]],
        dataset_ner_tags: list[list[int]],
        token_to_idx: dict[str, int],
        stats: dict[str, any]
):
    model.train()

    inputs = make_inputs(
        dataset_tokens,
        dataset_ner_tags,
        token_to_idx,
        compute_loss=True
    )
    output = model(**inputs)
    loss = output['loss']
    loss.backward()

    stats['loss'] += loss.item()
    stats['num_examples'] += len(dataset_tokens)
    

def print_eval(
        model,
        tokens: list[str],
        token_to_idx: dict[str, int],
        ner_tags: list[int],
        ner_tag_to_desc: Optional[dict[int, str]] = None,
        print_fn: Callable[[str], None] = print,
):
    model.eval()

    inputs = make_inputs(
        [tokens],
        [ner_tags],
        token_to_idx,
        decode_tags=True
    )
    viterbi_decode = model(**inputs)['tags']
    predictions = viterbi_decode[0][0]
    assert len(predictions) == len(ner_tags)

    if ner_tag_to_desc is None:
        ner_tag_to_desc = defaultdict(lambda x: str(x))

    print_fn('')
    print_fn(f'{"Tokens":<20} {"Pred":<7} {"Actual":<7}')
    for token, prediction, actual in zip(tokens, predictions, ner_tags):
        print_fn(f'{token:<20} {ner_tag_to_desc[prediction]:<7} {ner_tag_to_desc[actual]:<7}')
    print_fn('')
