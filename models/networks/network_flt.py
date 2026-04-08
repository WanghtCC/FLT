import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x
    
class FFN_conv(nn.Module):
    def __init__(self, dim):
        super(FFN_conv, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1, groups=self.dim_sp),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2, groups=self.dim_sp),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3, groups=self.dim_sp),
        )

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1(x[1])
        x[2] = self.conv2(x[2])
        x[3] = self.conv3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)

        return x
    
    def flops(self, x_size):
        H, W = x_size
        flops = 0
        flops += 2 * H * W * self.dim * self.dim * 2 * 1 * 1
        flops += 2 * H * W * self.dim_sp * self.dim_sp * 3 * 3 // self.dim_sp
        flops += 2 * H * W * self.dim_sp * self.dim_sp * 5 * 5 // self.dim_sp
        flops += 2 * H * W * self.dim_sp * self.dim_sp * 7 * 7 // self.dim_sp
        flops += 2 * H * W * self.dim * 2 * self.dim * 1 * 1

        return flops
    
class FourierBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super(FourierBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.idx_dict = {}
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)

        self.bn = nn.Sequential(
            nn.BatchNorm2d(dim * 2),
            nn.BatchNorm2d(dim * 2),
            nn.BatchNorm2d(dim * 2)
        )

        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature = (dim // num_heads) ** -0.5
        self.conv_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def softmax(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim=dim, keepdim=True) + 1)
        return logit

    def fft(self, x, n):
        x = torch.fft.rfft2(x.float(), norm='ortho')
        x = torch.cat([x.real, x.imag], dim=1)
        x = self.bn[n](x)
        b, c = x.shape[:2]
        x = x.contiguous().view(b, c, -1)
        return x
    def ifft(self, x, h, w):
        b, c = x.shape[:2]
        x = x.view(b, c, h, -1)
        x = torch.chunk(x, 2, dim=1)
        x = torch.complex(x[0], x[1])
        x = torch.fft.irfft2(x, s=(h, w), norm='ortho')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))

        # attn
        h, w = qkv.shape[2], qkv.shape[3]
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = self.fft(q, 0)
        k = self.fft(k, 1)
        v = self.fft(v, 2)
        
        q = rearrange(q, 'b (head c) hw -> b head c hw', head = self.num_heads)
        k = rearrange(k, 'b (head c) hw -> b head c hw', head = self.num_heads)
        v = rearrange(v, 'b (head c) hw -> b head c hw', head = self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax(attn, dim=-1)

        x = (attn @ v)
        x = rearrange(x, 'b head c hw -> b (head c) hw')
        x = self.ifft(x, h, w)
        x = self.conv_out(x)

        return x
    
    def flops(self, x_size):
        H, W = x_size
        flops = 0
        flops += 2 * H * W * self.dim * self.dim * 3 * 1 * 1
        flops += 2 * H * W * self.dim * 3 * self.dim * 3 * 3 * 3 // (self.dim * 3)
        flops += 3 * 2 * self.dim * 2 * H * W
        log2_HW = torch.log2(torch.tensor(H * W, dtype=torch.float32)).item()
        flops += 3 * self.dim * H * W * log2_HW * 2
        flops += self.dim * H * W * log2_HW * 2

        c = self.dim * 2 // self.num_heads
        flops += self.num_heads * (2 * H * W * c * c + 2 * H * W * c * c)
        flops += 2 * 6 * H * W * self.dim
        flops += 2 * self.dim * self.dim * 1 * 1 * H * W

        return flops
    
class LocalBlock(nn.Module):
    def __init__(self, dim):
        super(LocalBlock, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2

        self.conv1 = nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1, dilation=1, groups=self.dim_sp)
        self.conv2 = nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = torch.cat([x1, x2], dim=1)
        return x
    
    def flops(self, x_size):
        H, W = x_size
        flops = 0
        flops += 2 * self.dim_sp * self.dim_sp * 3 * 3 * H * W // self.dim_sp
        flops += 2 * self.dim_sp * self.dim_sp * 3 * 3 * H * W // self.dim_sp

        return flops
    
class FourierSwinTransformerMixerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, is_drop=True, mlp_type='ffn'):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.is_drop = is_drop

        self.norm1 = norm_layer(dim)
        self.conv_init = nn.Conv2d(dim, dim * 2, 1)
        # local
        self.local = LocalBlock(dim)
        # global
        self.freq = FourierBlock(dim, num_heads)

        self.gelu = nn.GELU()

        # fuse
        self.cam = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim * 2 // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2 // 2, dim * 2, 1),
            nn.Sigmoid()
        )

        self.conv_fina = nn.Conv2d(dim * 2, dim, 1)

        self.norm2 = norm_layer(dim)
        
        if is_drop:
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if mlp_type == 'ffn':
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif mlp_type == 'ffn_conv':
            self.mlp = FFN_conv(dim=dim)
        
    
    def forward(self, x):
        B, C, H, W = x.shape
        res = x
        
        x = self.norm1(x)
        x = self.conv_init(x)
        
        x = list(torch.split(x, self.dim, dim=1))
        x_local = self.local(x[0])# + x[0]

        x_global = self.freq(x[1]) + x[1]

        # fusion
        x = torch.cat([x_local, x_global], dim=1)
        x = self.gelu(x)
        x = self.cam(x) * x
        x = self.conv_fina(x)

        if self.is_drop:
            x = self.drop_path(x)
        x = x + res

        # ffn
        res = x
        if self.is_drop:
            x = self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.mlp(self.norm2(x))
        x = res + x

        return x
    
    def flops(self, x_size):
        H, W = x_size
        flops = 0
        flops += 4 * self.dim * H * W
        flops += 2 * self.dim * self.dim * 2 * 1 * 1 * H * W
        flops += self.local.flops(x_size)
        flops += self.freq.flops(x_size)
        flops += 2 * self.dim * H * W
        flops += 2 * self.dim * 2 * self.dim * 1 * 1 * 1 * 1
        flops += 2 * self.dim * 2 * self.dim * 1 * 1 * 1 * 1
        flops += 2 * self.dim * 2 * self.dim * 1 * 1 * H * W
        flops += 4 * self.dim * H * W
        flops += self.mlp.flops(x_size)
        return flops

class RDTB(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm, mlp_type='ffn'):
        super().__init__()
        self.dim = dim

        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = FourierSwinTransformerMixerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, is_drop=True, mlp_type=mlp_type
            )
            self.blocks.append(blk)
        
    def forward(self, x):
        res = x
        for blk in self.blocks:
            x = blk(x)
        return x + res
    
    def flops(self, x_size):
        flops = 0
        H, W = x_size
        for blk in self.blocks:
            flops += blk.flops(x_size)
        return flops
    
class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        self.scale = scale
        self.num_out_ch = num_out_ch
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        out_channels = (self.scale ** 2) * self.num_out_ch
        flops = H * W * self.num_feat * out_channels * 3 * 3 * 2
        return flops

class FLT(nn.Module):
    def __init__(self, img_size=64, in_chans=3, embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], mlp_ratio=4., drop_rate=0., drop_path_rate=0.1, norm_layer=nn.BatchNorm2d, upscale=2, img_range=1., upsampler='', **kwargs):
        super(FLT, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_size = img_size
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # deep feature extraction
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # merge non-overlapping patches into image
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RDTB(
                dim=self.embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                mlp_type='ffn_conv'  # ffn | ffn_conv
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer
        self.conv_after_body = nn.Conv2d(self.num_features, self.num_features, 3, 1, 1)
        
        # high quality image reconstruction
        self.upsample = UpsampleOneStep(upscale, self.num_features, num_out_ch, input_resolution=img_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        H, W = x_size

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        
        return x
    
    def forward(self, x):
        H, W = x.shape[2:]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
    
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
            
        x = self.upsample(x)

        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self):
        flops = 0
        if self.img_size.__class__ == int:
            x_size = to_2tuple(self.img_size)
        else:
            x_size = (self.img_size[0], self.img_size[1])
        H, W = x_size
        flops += H * W * 3 * self.embed_dim * 9 * 2
        for i, layer in enumerate(self.layers):
            flops += layer.flops(x_size)

        flops += H * W * self.embed_dim * self.embed_dim * 9 * 2
        
        flops += self.upsample.flops()
        return flops

if __name__ == '__main__':
    dim = 60
    upscale = 2
    window_size = 8
    height = 640 // upscale
    width = 480 // upscale
    model = FLT(upscale=upscale, 
                    img_size=(height, width),
                    window_size=window_size,
                    img_range=1., depths=[1, 1, 1, 1, 1, 1],
                    embed_dim=dim,
                    num_heads=[8, 8, 8, 8, 8, 8],
                    mlp_ratio=2)
    print(model)
    print(f"{model.flops() / 1e9} GFLOPs")
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6} M")
    x = torch.randn((1, 3, height, width))
    print(model(x).shape)
