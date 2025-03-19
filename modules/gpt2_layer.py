from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    TODO: Implement this helper method for the forward function.
      - This function is applied after the multi-head attention layer as well as after the feed forward layer.
      - GPT-2 layer applies dropout to the transformed output of each sub-layer,
        before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
        IN THIS FUNCTION.
    """
    """
        对于每个子层（多头注意力和前馈层）：
          1. 先对子层输出调用 dense_layer 做变换；
          2. 应用 dropout 防止过拟合；
          3. 再将变换后的输出和原始输入相加（残差连接）。
        """
    ### YOUR CODE HERE
    transformed = dense_layer(output)
    dropped = dropout(transformed)
    return input + dropped
    raise NotImplementedError


  def forward(self, hidden_states, attention_mask):
    """
    TODO: Implement the forward pass. Some key points to consider:
           - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
           - Layer normalization applied *before* the attention layer and feed-forward layer.
           - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
           - A feed-forward layer that applies transformations to further refine the hidden states.
    """
    """
    前向传播：
      1. 多头注意力子层：
          - 先对 hidden_states 应用 layer norm；
          - 使用 self_attention 计算自注意力；
          - 利用 add() 方法，将经过 attention_dense 及 dropout 后的结果与原始 hidden_states 相加。
      2. 前馈子层：
          - 对残差输出再进行 layer norm；
          - 经过一层全连接（self.interm_dense）和激活函数 gelu；
          - 再经过 self.out_dense 得到前馈子层的输出；
          - 使用 add() 方法，将 dropout 之后的前馈输出与输入相加。
    """

    ### YOUR CODE HERE

    # --- Multi-head Attention 子层 ---
    # 先对输入进行 layer norm
    attn_input = self.attention_layer_norm(hidden_states)
    # 计算自注意力（注意这里的 attention_mask 用于保证因果性）
    attn_output = self.self_attention(attn_input, attention_mask)
    # 通过 dense、dropout 后与原输入做残差连接
    hidden_states = self.add(hidden_states, attn_output, self.attention_dense, self.attention_dropout)

    # --- Feed-Forward 子层 ---
    # 对上一步的结果再次进行 layer norm
    ffn_input = self.out_layer_norm(hidden_states)
    # 经过前馈网络：先线性变换、激活，再线性变换得到前馈输出
    intermediate_output = self.interm_af(self.interm_dense(ffn_input))
    ffn_output = self.out_dense(intermediate_output)
    # 注意 feed-forward 分支这里不需要额外变换，因此使用 identity 函数作为 dense_layer
    hidden_states = self.add(hidden_states, ffn_output, lambda x: x, self.out_dropout)
    raise NotImplementedError

