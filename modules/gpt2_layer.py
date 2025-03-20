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
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size) #线性层
    self.interm_af = F.gelu  #激活函数
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
      TODO: 实现这个辅助方法用于前向传播函数。
      - 这个函数在多头注意力层和前馈层之后应用。
      - GPT-2 层在每个子层的变换输出上应用 dropout,然后再将其加到子层的输入上。我们**不在这个函数中应用层归一化**。
    """
    ### YOUR CODE HERE
    transformed = dense_layer(output)
    dropped = dropout(transformed)
    return input + dropped


  def forward(self, hidden_states, attention_mask):
    """
    TODO: Implement the forward pass. Some key points to consider:
           - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
           - Layer normalization applied *before* the attention layer and feed-forward layer.
           - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
           - A feed-forward layer that applies transformations to further refine the hidden states.
    """
    """
    TODO: 实现前向传播。需要考虑以下关键点：
      - 一个多头注意力层（CausalSelfAttention），该层基于带遮罩的输入计算自注意力。
      - 在注意力层和前馈层之前应用层归一化。
      - 按照作业中的流程图，应用 dropout、残差连接以及层归一化。（使用 self.add）
      - 一个前馈层，用于进一步调整隐藏状态。
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
    # ffn_output = self.out_dense(intermediate_output)
    # 注意 feed-forward 分支这里不需要额外变换，因此使用 identity 函数作为 dense_layer
    output = self.add(hidden_states, intermediate_output, self.out_dense, self.out_dropout)

    return output

