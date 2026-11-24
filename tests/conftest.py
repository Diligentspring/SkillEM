import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from skillem.model import HFModel
from skillem.schema import FIELDS, Task, Step


@pytest.fixture
def tiny_model():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    vocab = {'<unk>': 0, '<eos>': 1, '<pad>': 2, 'User': 3, ':': 4, 'Assistant': 5,
             '0': 6, '1': 7, 'answer': 8, 'Paris': 9, '<': 10, '>': 11, '/': 12, 'search': 13}
    t = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
    t.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token='<unk>', eos_token='<eos>', pad_token='<pad>')
    config = GPT2Config(vocab_size=len(vocab), n_positions=2048, n_ctx=2048,
                       n_embd=16, n_layer=1, n_head=2, eos_token_id=1, pad_token_id=2)
    config._attn_implementation = "eager"
    model = GPT2LMHeadModel(config)
    return HFModel(model, tok, max_field_tokens=2, max_action_tokens=8, max_context=2048)


@pytest.fixture
def task():
    return Task('audit', '0', ['Paris'], 'nq', 'train',
                steps=[Step('<search>0</search>', '<information>1</information>'),
                       Step('<answer>Paris</answer>')], verified=True)
