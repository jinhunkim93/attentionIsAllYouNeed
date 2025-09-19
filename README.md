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
We apply dropout [33] to the output of each sub-layer, before it is added to the
sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
Pdrop = 0.1.

Input Variables

Embedding (Hidden Layer) Number : number of values to represent a token, also the number of activation functions applied in word embedding process
Batch Size : Number of sequences processed per batch, also the number of Samples to look at before updating.

Number of Encoder Layers : Number of layers used for the Encoder
Number of Decoder Layers : Number of layers used for the Decoder

Number of Encoder Attention Heads : Number of Heads for the Encoder MultiHeadAttention Layer
Number of Decoder Attention Heads : Number of Heads for the Decoder MultiHeadAttention Layer

Encoder DropOut Probability : Probability value to set a value to zero
Decoder DropOut Probability : Probability value to set a value to zero

Position Wise Feed ForwardInner Layer Inner Layer Dimension : Original paper set this value to 2048