import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many
def exists(val):
    return val is not None
def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )
class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
    def forward(self, x, latents, device = None):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        original_dtype = torch.float
        if self.norm_media.weight.dtype == torch.bfloat16:
            self.norm_media.float()
            # original_dtype = torch.bfloat16
        x = self.norm_media(x).to(device=device if device is not None else x.device)
        if self.norm_latents.weight.dtype == torch.bfloat16:
            self.norm_latents.float()
        latents = self.norm_latents(latents).to(device=device if device is not None else latents.device)

        # latents = latents.to(dtype=original_dtype)
        # x = x.to(dtype=original_dtype)
        # latents = latents.to(dtype=torch.bfloat16)
        # x = x.to(dtype=torch.bfloat16)

        b, m, h = *x.shape[:2], self.heads
        q = self.to_q(latents)
        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)
        q = q * self.scale
        # attention
        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)
class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        
        use_inner_query = False,
        num_latents = 30
    ):
        super().__init__()
        self.use_inner_query = use_inner_query
        if use_inner_query:
            self.latents = nn.Parameter(torch.randn(num_latents, dim))
            # self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))
        self.norm = nn.LayerNorm(dim)
    def forward(self, latents=None, x=None, x_posi=None, device=None):
        assert (self.use_inner_query and latents == None) or (not self.use_inner_query and latents != None), ''
        b,n,d = x.shape
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')
            if x_posi is not None:
                x_posi = rearrange(x_posi, 'b n d -> b 1 n d')
        
        if not self.use_inner_query:
            if latents.ndim == 3:
                latents = rearrange(latents, 'b n d -> b 1 n d')
        else:
            # latents = rearrange(self.latents, 'n d -> b 1 n d')
            latents = repeat(self.latents, 'n d -> b m n d', b = b, m = 1)
        # times = x.shape[1]
        # x = x + self.media_pos_emb[:times]
        if x_posi is not None:
            x = x + x_posi
        # import pdb;pdb.set_trace()
        # latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])
        for attn, ff in self.layers:
            latents = attn(x, latents).to(device = device if device is not None else latents.device) + latents
            latents = ff(latents).to(device = device if device is not None else latents.device) + latents
        return self.norm(latents)
# gated cross attention
class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        only_attend_immediate_media = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        # whether for text to only attend to immediate preceding image, or all images
        self.only_attend_immediate_media = only_attend_immediate_media
    def forward(
        self,
        x,
        media,
        media_locations = None
    ):
        b, t, m = media.shape[:3]
        h = self.heads
        x = self.norm(x)
        q = self.to_q(x)
        media = rearrange(media, 'b t n d -> b (t n) d')
        k, v = self.to_kv(media).chunk(2, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)
        q = q * self.scale
        sim = einsum('... i d, ... j d -> ... i j', q, k)
        if exists(media_locations):
            text_time = media_locations.cumsum(dim = -1) # at each boolean of True, increment the time counter (relative to media time)
            media_time = torch.arange(t, device = x.device) + 1
            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge
            text_to_media_mask = mask_op(rearrange(text_time, 'b i -> b 1 i 1'), repeat(media_time, 'j -> 1 1 1 (j m)', m = m))
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
            attn = attn.masked_fill(text_without_media_mask, 0.)
        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        only_attend_immediate_media = True
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(dim = dim, dim_head = dim_head, heads = heads, only_attend_immediate_media = only_attend_immediate_media)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))
        self.ff = FeedForward(dim, mult = ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))
    def forward(
        self,
        x,
        media,                  # media tensor, encoded by perceiver resample - (batch, time, latents, dim)
        media_locations = None  # boolean tensor indicating positions of media - (batch, sequence)
    ):
        x = self.attn(x, media, media_locations = media_locations) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh()  + x
        return x
if __name__ == "__main__":
    # 随机参数
    batch_size = 2
    num_input_tokens = 50
    dim = 128
    num_latents = 30
    depth = 2
    # 创建输入
    x = torch.randn(batch_size, num_input_tokens, dim)
    # ----- use_inner_query = True -----
    model = PerceiverResampler(
        dim=dim,
        depth=depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        use_inner_query=True,
        num_latents=num_latents
    )
    print("Testing with use_inner_query=True")
    # 按接口，只要给 x
    out = model(x=x)
    print("Input shape (x):", x.shape)
    print("Output shape (out):", out.shape)
    # out 的 shape 应该是 [batch_size, 1, num_latents, dim]
    assert out.shape == (batch_size, 1, num_latents, dim), f"Output shape mismatch: expected ({batch_size}, 1, {num_latents}, {dim}) got {out.shape}"
    # ----- use_inner_query = False -----
    model2 = PerceiverResampler(
        dim=dim,
        depth=depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        use_inner_query=False,
        num_latents=num_latents
    )
    print("\nTesting with use_inner_query=False")
    # 也随机造一个 latents
    latents = torch.randn(batch_size, num_latents, dim)
    out2 = model2(latents=latents, x=x)
    print("Input shape (x):", x.shape)
    print("Input shape (latents):", latents.shape)
    print("Output shape (out):", out2.shape)
    assert out2.shape == (batch_size, 1, num_latents, dim), f"Output shape mismatch: expected ({batch_size}, 1, {num_latents}, {dim}) got {out2.shape}"
    print("\nBoth use_inner_query True and False tests passed.")
    """
        Testing with use_inner_query=True
        Input shape (x): torch.Size([2, 50, 128]) b n d 
        Output shape (out): torch.Size([2, 1, 30, 128]) b t n d
        Testing with use_inner_query=False
        Input shape (x): torch.Size([2, 50, 128]) b n d
        Input shape (latents): torch.Size([2, 30, 128]) b n d 
        Output shape (out): torch.Size([2, 1, 30, 128]) b t n d
    """

