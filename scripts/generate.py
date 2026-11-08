"""Entry point for autoregressive text generation."""


import torch
from tokenizers import Tokenizer
from minigpt.model import GPTLanguageModel

CHECKPOINT_PATH = "artifacts/checkpoints/minigpt_step0100.pt"
TOKENIZER_PATH = "artifacts/tokenizer.json"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate(model, input_ids, max_new_tokens=100, temperature=1.0, eos_token_id=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):

            logits, _ = model(input_ids)
            next_token_logits = logits[:, -1, :]

            probs = torch.softmax(
                next_token_logits / temperature,
                dim=-1
            )

            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

            if eos_token_id is not None:
                if torch.all(next_token==eos_token_id):
                    break

    return input_ids




def main() -> None:
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model = GPTLanguageModel(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt = "Hello, "
    input_ids = tokenizer.encode(prompt).ids

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device
    )

    eos_token_id = tokenizer.token_to_id("<eos>")
    generated_ids = generate(
        model,
        input_ids,
        max_new_tokens=100,
        temperature=1.2,
        eos_token_id=eos_token_id
    )

    generated_text = tokenizer.decode(
        generated_ids[0].tolist(),
        skip_special_tokens=True
    )

    print(generated_text)





if __name__ == "__main__":
    main()
