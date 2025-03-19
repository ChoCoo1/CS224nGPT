import math
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    """
    Implements multi-head attention computation.
    
    key, query, value: shape [bs, num_attention_heads, seq_len, attention_head_size]
    attention_mask: shape [bs, 1, 1, seq_len]
    
    Returns the attended values after applying attention mechanism.
    """
    d_k = query.size(-1)  # attention_head_size
    
    # Calculate the attention scores (scaled dot-product attention)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # [bs, num_attention_heads, seq_len, seq_len]
    
    # Apply the attention mask (make sure the attention only attends to valid tokens)
    if attention_mask is not None:
      scores = scores.masked_fill(attention_mask == 0, -1e9)
    
    # Apply softmax to get the attention weights (probabilities)
    attn_weights = F.softmax(scores, dim=-1)  # [bs, num_attention_heads, seq_len, seq_len]
    
    # Apply dropout on attention weights (to prevent overfitting)
    attn_weights = self.dropout(attn_weights)
    
    # Multiply the attention weights with the value tensor to get the attended output
    attn_output = torch.matmul(attn_weights, value)  # [bs, num_attention_heads, seq_len, attention_head_size]
    
    # Rearrange the output to have shape [bs, seq_len, hidden_size]
    attn_output = rearrange(attn_output, 'b h t d -> b t (h d)')  # [bs, seq_len, hidden_size]
    return attn_output

  def forward(self, hidden_states, attention_mask):
      """
      hidden_states: [bs, seq_len, hidden_state]
      attention_mask: [bs, 1, 1, seq_len]
      output: [bs, seq_len, hidden_state]
      """
      # Generate the key, value, query for each token using the transform function.
      # The size of key_layer, value_layer, query_layer is [bs, num_attention_heads, seq_len, attention_head_size].
      key_layer = self.transform(hidden_states, self.key)
      value_layer = self.transform(hidden_states, self.value)
      query_layer = self.transform(hidden_states, self.query)
      
      # Calculate the multi-head attention output.
      attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
      return attn_value