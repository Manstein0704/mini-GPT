"""BPE tokenizer training, save/load, encode/decode exercises."""
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing
from minigpt.data import load_tinystories
from pathlib import Path

def main():
    output_path = Path("artifacts/tokenizer.json")
    if output_path.exists():
        print("Loading existing tokenizer...")
        tokenizer = Tokenizer.from_file(str(output_path))
    else:
        print("Training new tokenizer ...")

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


        def text_iterator():
            for example in train_ds:
                yield example["text"]


        tokenizer.train_from_iterator(
            text_iterator(),
            trainer=trainer,
            length=len(train_ds)
        )

        tokenizer.decoder = decoders.ByteLevel()
        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")

        assert bos_id is not None, "<bos> token is missing"
        assert eos_id is not None, "<eos> token is missing"

        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[
                ("<bos>", bos_id),
                ("<eos>", eos_id)
            ]
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_path))


    # Verify tokenizer
    text = "Once upon a time, there was a little girl."

    encoded = tokenizer.encode(text)

    print("Original:", text)
    print("Tokens:", encoded.tokens)
    print("Token IDs:", encoded.ids)
    print("Decoded:", tokenizer.decode(encoded.ids))
    print("Vocabulary size:", tokenizer.get_vocab_size())

    assert tokenizer.decode(encoded.ids) == text
    assert tokenizer.get_vocab_size() == 8000

    print("Tokenizer test passed!")

if __name__ == "__main__":
    main()



