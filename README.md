# attentionIsAllYouNeed
Implement the basic architecture illustrated in the paper "Attention is All You Need" to understand the Transformer architecture.

The paper describes the Transformer architecture as follows:

Data Preparation: Input Tokens (BPE Encoding) -> Input Embedding -> Position Encoding

1. Encoder
    1) Consists of # Encoder layers
    2) Each Encoder layer consists of a multiHeadAttention Sublayer and FeedForward Sublayer. Layer Normalization and Dropout is applied to the output of each Sublayer.


2. Decoder
    1) Consists of # Decoder layers
    2) Each Decoder layer consists of a multiHeadAttention (masked) Sublayer, another multiHeadAttention Sublayer, and a FeedForward Sublayer. Layer Normalization and Dropout is applied to the output of each Sublayer.
    Note: the Encoder input value for the EncoderDecoder Attention calculation comes only from the final Encoder Layer.



* layerNormalization, dropout
Batch Size: Number of Samples to look at before updating

Number of Tokens : 
Embedding (Hidden Layer) Number : number of 
Batch Size : Number of 

Number of Encoder Layers : Number of layers used for the Encoder
Number of Decoder Layers : Number of layers used for the Decoder

Number of Encoder Attention Heads : Number of Heads for the Encoder MultiHeadAttention Layer
Number of Decoder Attention Heads : Number of Heads for the Decoder MultiHeadAttention Layer

Encoder DropOut Probability : Probability value to set a value to zero
Decoder DropOut Probability : Probability value to set a value to zero