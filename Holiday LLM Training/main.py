import torch
from model import GPTLanguageModel

# load checkpoint of the model
checkpoint = torch.load("mini_ldr_gpt.pth", map_location="cpu")

vocab_size = checkpoint["vocab_size"]
block_size = checkpoint["block_size"]
n_embd = checkpoint["n_embd"]
n_head = checkpoint["n_head"]
n_layer = checkpoint["n_layer"]
itos = checkpoint["itos"]
stoi = checkpoint["stoi"]

# rebuild model and add weights
model = GPTLanguageModel(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=0.0,   # no dropout at inference
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# encoding/decoding
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] for i in l])

# user prompt
while True:
    prompt = input("\nEnter a prompt (or 'quit'): ")
    if prompt.lower() == "quit":
        break

    context = torch.tensor([encode(prompt)], dtype=torch.long)
    output = model.generate(context, max_new_tokens=300)

    print("\n--- Generated text ---")
    print(decode(output[0].tolist()))
