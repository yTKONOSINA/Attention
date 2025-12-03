import matplotlib.pyplot as plt

def visualize_attention(attention_weights, tokens):
    """
        Visualize the attention weights for the first sentence.
    """

    num_layers = len(attention_weights)
    num_heads = len(attention_weights[0].tensor[0])

    fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 4, num_layers * 4))
    
    # Filter out padding tokens
    real_tokens = [t for t in tokens if t != "[PAD]"]
    seq_len = len(real_tokens)

    for layer_idx, layer_attn in enumerate(attention_weights):
        heads_attn = layer_attn.tensor[0]
        
        for head_idx in range(num_heads):
            ax = axes[layer_idx][head_idx]
            full_matrix = heads_attn[head_idx]
            attn_matrix = [row[:seq_len] for row in full_matrix[:seq_len]]
            
            cax = ax.imshow(attn_matrix, cmap='Reds', aspect='auto')
            
            ticks = range(seq_len)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            
            ax.xaxis.tick_top()
            
            ax.set_xticklabels(real_tokens, rotation=90)
            ax.set_yticklabels(real_tokens)
            ax.set_title(f"Layer {layer_idx} Head {head_idx}", pad=40)
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04) # Legend
            
    plt.tight_layout()
    plt.show()
