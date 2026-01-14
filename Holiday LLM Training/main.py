import torch
from model import GPTLanguageModel

# --- optional device (GPU if available) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- load checkpoint ---
checkpoint = torch.load("mini_ldr_gpt.pth", map_location=device)

vocab_size = checkpoint["vocab_size"]
block_size = checkpoint["block_size"]
n_embd = checkpoint["n_embd"]
n_head = checkpoint["n_head"]
n_layer = checkpoint["n_layer"]
itos = checkpoint["itos"]
stoi = checkpoint["stoi"]

# --- rebuild model and load weights ---
model = GPTLanguageModel(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=0.0,   # no dropout at inference
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# --- encoding/decoding functions ---
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] for i in l])

# --- user prompt loop ---
while True:
    prompt = input("\nEnter a prompt (or 'quit'): ")
    if prompt.lower() == "quit":
        break

    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    output = model.generate(context, max_new_tokens=300)

    print("\n--- Generated text ---")
    print(decode(output[0].tolist()))
