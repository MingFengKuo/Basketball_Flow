import torch

from torch import nn
from timm.models.vision_transformer import PatchEmbed, Block

class Extractor(nn.Module):
    def __init__(self, z_dim=384, f_dim=28, t_dim=120, patch_num=24, depth=1):
        super().__init__()
        self.f_dim = f_dim
        self.t_dim = t_dim
        self.z_dim = z_dim
        # param
        self.embed_dim = 384
        self.patch_num = patch_num
        
        # model
        self.patch_embed = PatchEmbed(img_size=(self.t_dim, self.f_dim), 
                                      patch_size=(self.t_dim // self.patch_num, self.f_dim), 
                                      in_chans=1, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_num + 1, self.embed_dim))
        self.blocks = nn.ModuleList([Block(dim=self.embed_dim, num_heads=6, mlp_ratio=2., qkv_bias=True)
                                     for i in range(depth)])

        self.norm = nn.LayerNorm(self.embed_dim)
        self.linear = nn.Linear(self.embed_dim, self.z_dim)

    def forward(self, z):
        # reshape
        z = z.view(-1, 1, self.t_dim, self.f_dim)
        # embed patches
        z = self.patch_embed(z)
        # append cls token
        cls_tokens = self.cls_token.expand(z.shape[0], -1, -1)
        z = torch.cat((cls_tokens, z), dim=1)
        # add pos embed
        z = z + self.pos_embed
        # apply Transformer blocks
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        z = self.linear(z)
        z = z[:, 0]
        # return
        return z

class Encoder(nn.Module):
    def __init__(self, z_dim=384, f_dim=28, t_dim=120, patch_num=24, depth=1):
        super().__init__()
        self.f_dim = f_dim
        self.t_dim = t_dim
        self.z_dim = z_dim
        # param
        self.embed_dim = 384
        self.patch_num = patch_num
        
        # model
        self.patch_embed = PatchEmbed(img_size=(self.t_dim, self.f_dim), 
                                      patch_size=(self.t_dim // self.patch_num, self.f_dim), 
                                      in_chans=1, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_num + 1, self.embed_dim))
        self.blocks = nn.ModuleList([Block(dim=self.embed_dim, num_heads=6, mlp_ratio=2., qkv_bias=True)
                                     for i in range(depth)])
        
        self.mean_block  = Block(dim=self.embed_dim, num_heads=6, mlp_ratio=2., qkv_bias=True)
        self.mean_norm   = nn.LayerNorm(self.embed_dim)
        self.mean_linear = nn.Linear(self.embed_dim, self.z_dim)
        
        self.var_block  = Block(dim=self.embed_dim, num_heads=6, mlp_ratio=2., qkv_bias=True)
        self.var_norm   = nn.LayerNorm(self.embed_dim)
        self.var_linear = nn.Linear(self.embed_dim, self.z_dim)
        
    def forward(self, z):
        # reshape (for ViT)
        z = z.view(-1, 1, self.t_dim, self.f_dim)
        # embed patches
        z = self.patch_embed(z)
        # append cls token
        cls_tokens = self.cls_token.expand(z.shape[0], -1, -1)
        z = torch.cat((cls_tokens, z), dim=1)
        # add pos embed
        z = z + self.pos_embed
        # apply Transformer blocks
        for blk in self.blocks:
            z = blk(z)
        # branch (mean)
        z_mu = self.mean_block(z)
        z_mu = self.mean_norm(z_mu)
        z_mu = self.mean_linear(z_mu)
        z_mu = z_mu[:, 0]
        # branch (var)
        z_var = self.var_block(z)
        z_var = self.var_norm(z_var)
        z_var = self.var_linear(z_var)
        z_var = z_var[:, 0]
        # return
        return z_mu, z_var

class Generator(nn.Module):
    def __init__(self, z_dim=384, f_dim=28, t_dim=120, patch_num=24, depth=1):
        super().__init__()
        self.f_dim = f_dim
        self.t_dim = t_dim
        self.z_dim = z_dim
        # param
        self.embed_dim = 384
        self.patch_num = patch_num
        
        # model
        self.patch_embed = nn.Linear(self.z_dim, self.patch_num * self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_num, self.embed_dim))
        self.blocks = nn.ModuleList([Block(dim=self.embed_dim, num_heads=6, mlp_ratio=2., qkv_bias=True)
                                     for i in range(depth)])
        
        self.norm = nn.LayerNorm(self.embed_dim)
        self.linear = nn.Linear(self.embed_dim, (self.t_dim//self.patch_num) * self.f_dim)
    
    def forward(self, z):
        # reshape (to latent)
        z = self.patch_embed(z)
        z = z.view(-1, self.patch_num, self.embed_dim)
        # apply Transformer blocks
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        z = self.linear(z)
        z = z.view(-1, self.t_dim, self.f_dim)
        fake_x = torch.cat((z[:, :, :22], nn.Sigmoid()(z[:, :, 22:])), axis=-1) 
        return fake_x

class Discriminator(nn.Module):
    def __init__(self, z_dim=384, c_dim=18, f_dim=28, t_dim=120, patch_num=24, depth=1):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.t_dim = t_dim
        self.z_dim = z_dim
        # param
        self.embed_dim = 384
        self.patch_num = patch_num
        # overlap param
        self.padding_size = 2
        self.unfold_size = self.padding_size * 2 + (self.t_dim // self.patch_num)
        
        # model
        self.patch_embed = PatchEmbed(img_size=(self.unfold_size * self.patch_num, self.f_dim),
                                      patch_size=(self.unfold_size, self.f_dim),
                                      in_chans=1, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_num + 1, self.embed_dim))
        self.blocks = nn.ModuleList([
            Block(dim=self.embed_dim, num_heads=6, mlp_ratio=2., qkv_bias=True)
            for i in range(depth)])
        
        # model
        self.norm = nn.LayerNorm(self.embed_dim)
        self.score_linear = nn.Linear(self.embed_dim, 1)
        self.sketch_linear = nn.Linear(self.embed_dim, (self.t_dim//self.patch_num) * self.c_dim)
        
    def forward(self, x):
        # generate overlapping patches
        x = torch.cat((x[:, :self.padding_size], x, x[:, -self.padding_size:]), axis=1)
        x = x.unfold(dimension=1, size=self.unfold_size, step=self.t_dim // self.patch_num)
        x = x.transpose(3, 2)
        # reshape
        x = x.reshape(-1, 1, self.unfold_size * self.patch_num, self.f_dim)
        # embed patches
        x = self.patch_embed(x)
        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # add pos embed
        x = x + self.pos_embed
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # features
        features = x.reshape(-1, (self.patch_num + 1) * self.embed_dim)
        # scores
        scores = self.score_linear(x)[:, 0]
        # sketch
        sketch = self.sketch_linear(x)[:, 1:]
        sketch = sketch.view(-1, self.t_dim, self.c_dim)
        sketch = torch.cat((sketch[:, :, :12], nn.Sigmoid()(sketch[:, :, 12:])), axis=-1) 
        return features, scores, sketch