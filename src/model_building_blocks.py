import torch.nn as nn


class PatchEmbedding(nn.Module):
  """Converts an input image into a sequence of learnable embedding vector.

  Args:
    in_channels (int): Number of color channels for the input image.
    patch_size (int): Input image is divided into smaller patches of the given size.
    embedding_dim (int): Size of embedding to turn image into.
  """
  def __init__(self,
               in_channels:int=3,
               patch_size:int=4,
               embedding_dim:int=512):
    super().__init__()

    self.patch_size = patch_size

    # Layer to turn image into patches
    self.generate_patches = nn.Conv2d(in_channels=in_channels,
                                      out_channels=embedding_dim,
                                      kernel_size=patch_size,
                                      stride=patch_size,
                                      padding=0)
    
    # Layer to flatten the patches into a single dimension
    self.flatten = nn.Flatten(start_dim=2,
                              end_dim=3)
    
    # Define the forward method
  def forward(self, x):
    # check if the input size is divisible by patch size.
    img_res = x.shape[-1]
    assert img_res % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {img_res}, patch size: {self.patch_size}"
    x_patched = self.generate_patches(x)
    x_flattened = self.flatten(x_patched)
    # Make sure the output shape has the right order of dimensions i.e. [batch_size, P^2.C, N] -> [batch_size, N, P^2.C]
    return x_flattened.permute(0, 2, 1)

# define a class for MSA layer
class MultiheadAttention(nn.Module):
  def __init__(self,
               emb_dim:int=512,
               num_heads:int=8,
               attn_dropout:float=0):
    super().__init__()
    # Create normalization layer
    self.layer_norm = nn.LayerNorm(normalized_shape=emb_dim)
    # Create Multihead self attention layer
    self.msa = nn.MultiheadAttention(embed_dim=emb_dim,
                                     num_heads=num_heads,
                                     dropout=attn_dropout,
                                     batch_first=True)
  # Create a forward method
  def forward(self, x):
    x = self.layer_norm(x)
    attn_out, _ = self.msa(query=x,
                           key=x,
                           value=x,
                           need_weights=False) # if set to True, returns attention output weights
    return attn_out     

# Create a class for MLP block
class MLP(nn.Module):
  def __init__(self,
               emb_dim:int=512, 
               mlp_size:int=512,
               dropout:float=0.1): 
    super().__init__()

    # Create norm layer
    self.layer_norm = nn.LayerNorm(normalized_shape=emb_dim)

    # Create MLP layers
    self.mlp = nn.Sequential(
        nn.Linear(in_features=emb_dim,
                  out_features=mlp_size),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=mlp_size,
                  out_features=emb_dim),
        nn.Dropout(p=dropout)
    )
  def forward(self, x):
    x = self.layer_norm(x)
    x = self.mlp(x)
    return x
    

class ViTEncoder(nn.Module):
  def __init__(self,
               emb_dim:int=512, 
               num_heads:int=8, 
               mlp_size:int=512, 
               mlp_dropout:float=0.1,
               attn_dropout:float=0.):
    super().__init__()
    self.msa_layer = MultiheadAttention(emb_dim=emb_dim,
                                        num_heads=num_heads,
                                        attn_dropout=attn_dropout)
    self.mlp_layer = MLP(emb_dim=emb_dim,
                         mlp_size=mlp_size,
                         dropout=mlp_dropout)
    
  def forward(self, x):
    # Create residual connection for MSA and MLP layers by adding the input to the output
    x = self.msa_layer(x) + x
    x = self.mlp_layer(x) + x
    return x


