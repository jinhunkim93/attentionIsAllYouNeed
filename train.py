import torch
from transformer import Transformer

SRC = torch.randint(0, 10000, (32, 20))  # Example source batch (batch_size=32, seq_len=20)
TRG = torch.randint(0, 10000, (32, 20))  # Example target batch (batch_size=32, seq_len=20)

# Hyperparameters
D_MODEL = 512 # Model dimension for embeddings and all sub-layers
ENC_D_FF = 2048 # Feed-forward dimension for position-wise feed-forward networks
DEC_D_FF = 2048 # Feed-forward dimension for position-wise feed-forward networks

BATCH_SIZE = 64

ENC_NUM_HEADS = 8 # Number of encoder attention heads
ENC_NUM_LAYERS = 6 # Number of encoder layers
DEC_NUM_HEADS = 8 # Number of decoder attention heads
DEC_NUM_LAYERS = 6 # Number of decoder layers

SRC_VOCAB_SIZE = 10000  # Example source vocabulary size
TGT_VOCAB_SIZE = 10000  # Example target vocabulary size
MAX_SEQ_LEN = 100  # Maximum sequence length


DROPOUT = 0.1
LEARNING_RATE = 0.0001
NUM_EPOCHS = 20

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Transformer(
    src=SRC,
    trg=TRG,
    d_model=D_MODEL,
    enc_num_heads=ENC_NUM_HEADS,
    enc_num_layers=ENC_NUM_LAYERS,
    enc_d_ff=ENC_D_FF,
    dec_num_heads=DEC_NUM_HEADS,
    dec_num_layers=DEC_NUM_LAYERS,
    dec_d_ff=DEC_D_FF,
    dropout=DROPOUT
).to(DEVICE)

# 5.3 Optimizer : Adam with β1=0.9, β2=0.98 and ε=10−9
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # Assuming 0 is the padding index
