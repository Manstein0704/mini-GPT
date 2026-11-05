"""BPE tokenizer training, save/load, encode/decode exercises."""
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing
from data import load_tinystories


VOCAB_SIZE = 8000
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


tokenizer = Tokenizer(
    models.BPE(unk_token="<unk>")
)

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False
)

trainer = trainers.BpeTrainer(
    vocab_size = VOCAB_SIZE,
    special_tokens = SPECIAL_TOKENS,
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
)


train_ds = load_tinystories(split="train")


"""Train a byte-level BPE tokenizer on TinyStories."""
def text_iterator():
    for example in train_ds:
        yield example["text"]


tokenizer.train_from_iterator(
    text_iterator(),
    trainer=trainer,
    length=len(train_ds)
)

