import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Residual, Attention, PreNorm, LeFF, FeedForward, LCAttention

class TransformerLeFF(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, scale = 4, depth_kernel = 3, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, LeFF(dim, scale, depth_kernel)))
            ]))
    def forward(self, x):
        c = list()
        for attn, leff in self.layers:
            x = attn(x)
            cls_tokens = x[:, 0]
            c.append(cls_tokens)
            x = leff(x[:, 1:])
            x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1) 
        return x, torch.stack(c).transpose(0, 1)



class LCA(nn.Module):
    # I remove Residual connection from here, in paper author didn't explicitly mentioned to use Residual connection, 
    # so I removed it, althougth with Residual connection also this code will work.
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.ModuleList([
                PreNorm(dim, LCAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x
        


  
class CeiT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim = 192, depth = 12, heads = 3, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., 
                 emb_dropout = 0., conv_kernel = 7, stride = 2, depth_kernel = 3, pool_kernel = 3, scale_dim = 4, with_lca = False):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # IoT
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, conv_kernel, stride, 4),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(pool_kernel, stride)    
        )
        
        feature_size = image_size // 4
        assert feature_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size // patch_size) ** 2
        patch_dim = 32 * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerLeFF(dim, depth, heads, dim_head, scale_dim, depth_kernel, dropout)
        
        self.with_lca = with_lca
        if with_lca:
            self.LCA = LCA(dim, 4, 48, 384)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.conv(img)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        x, c = self.transformer(x)
        
        if self.with_lca:
            x = self.LCA(c)[:, -1]
        else:    
            x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
    
    

if __name__ == "__main__":
    
    img = torch.ones([1, 3, 224, 224])
    
    model = CeiT(224, 4, 100)
    
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]
    
    model = CeiT(224, 4, 100, with_lca = True)
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]