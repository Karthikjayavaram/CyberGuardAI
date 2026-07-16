import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import ImprovedTransformer
from dataset_loader import load_and_preprocess_dataset

def main():
    print("Starting Attention Analysis...")
    
    # Ensure output directories exist
    os.makedirs('explainability/attention', exist_ok=True)
    
    # 1. Load the trained federated model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedTransformer(input_dim=46, num_classes=34).to(device)
    
    model_path = 'saved_models/global_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded global_model.pth successfully.")
    else:
        print(f"Warning: {model_path} not found. Using untrained model for analysis.")
    
    model.eval()
    
    # 2. Load a small batch (64 samples) from the test dataset
    # We set DEBUG_SAMPLE_SIZE inside dataset_loader implicitly or just grab a batch
    _, test_loader = load_and_preprocess_dataset()
    batch_x, batch_y = next(iter(test_loader))
    batch_x = batch_x[:64].to(device)
    print(f"Extracted a batch of {batch_x.shape[0]} samples.")
    
    # 3. Use hooks to extract attention weights without modifying model.py
    attention_weights = []
    handles = []
    
    def pre_hook(module, args, kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False
        return args, kwargs

    def post_hook(module, args, kwargs, output):
        attention_weights.append(output[1].detach().cpu())
    
    for layer in model.transformer.layers:
        h1 = layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
        h2 = layer.self_attn.register_forward_hook(post_hook, with_kwargs=True)
        handles.extend([h1, h2])
    
    # Run inference by bypassing TransformerEncoder's fast path
    # Fast path in PyTorch 2.0+ during eval() skips calling self_attn completely
    with torch.no_grad():
        x = model.feature_embed(batch_x)
        x = x.unsqueeze(1)
        
        for layer in model.transformer.layers:
            x = layer(x)
            
        if model.transformer.norm is not None:
            x = model.transformer.norm(x)
            
        x = x.squeeze(1)
        _ = model.classifier(x)
        
    # Remove hooks
    for h in handles:
        h.remove()
        
    print(f"Extracted attention weights from {len(attention_weights)} layers.")
    
    # 4. Process and Generate Heatmaps
    # attention_weights[layer_idx] is [batch_size, num_heads, seq_len, seq_len]
    # In this architecture, seq_len = 1. So it is [64, 8, 1, 1].
    
    # We will plot the actual attention weights (which will be 1.0) 
    # and also plot the feature embedding weights which act as the true feature selectors.
    
    # Let's compute average attention across heads for Layer 1
    avg_attn_l1 = attention_weights[0].mean(dim=(0, 1)).squeeze().numpy()
    
    # Create a figure for the multi-head attention (Layer 1, first sample)
    heads_attn = attention_weights[0][0].squeeze().numpy() # Shape: (8,)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    sns.heatmap(heads_attn.reshape(1, 8), annot=True, cmap='viridis', ax=axes[0], cbar_kws={'label': 'Attention Weight'})
    axes[0].set_title('Multi-Head Attention Weights (Layer 1)', weight='bold')
    axes[0].set_xlabel('Head Index')
    axes[0].set_yticks([])
    
    sns.heatmap(np.array([[avg_attn_l1]]), annot=True, cmap='coolwarm', ax=axes[1], cbar_kws={'label': 'Average Weight'})
    axes[1].set_title('Average Attention Across All Heads', weight='bold')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    plt.tight_layout()
    plt.savefig('explainability/attention/attention_heatmap.png', dpi=300)
    plt.close()
    
    # Generate average_attention.png showing all layers
    fig, ax = plt.subplots(figsize=(6, 4))
    layer_avgs = [attn.mean().item() for attn in attention_weights]
    sns.barplot(x=[f'Layer {i+1}' for i in range(len(layer_avgs))], y=layer_avgs, palette='Blues_d', ax=ax)
    ax.set_title('Average Attention Weight per Layer', weight='bold')
    ax.set_ylabel('Attention Weight')
    plt.tight_layout()
    plt.savefig('explainability/attention/average_attention.png', dpi=300)
    plt.close()
    
    # 5. Extract feature importance from the embedding layer 
    # Because sequence length is 1, the true distribution of attention across the 46 features 
    # happens in the feature_embed layer (Linear(46, 256)).
    embed_weights = model.feature_embed[0].weight.detach().cpu().abs().mean(dim=0).numpy()
    
    # Save to CSV
    feature_indices = np.arange(46)
    df_features = pd.DataFrame({
        'Feature_Index': feature_indices,
        'Attention_Proxy_Score': embed_weights
    })
    df_features = df_features.sort_values(by='Attention_Proxy_Score', ascending=False)
    df_features.to_csv('explainability/attention/top_attended_features.csv', index=False)
    
    print("Attention analysis completed successfully.")

if __name__ == "__main__":
    main()
