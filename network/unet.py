import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_ch, num_groups):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, num_groups, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch, num_groups)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, alpha, beta, input_size=32, input_ch=3):
        super().__init__()
        num_res_blocks, ch = 2*alpha, 16*beta
        num_groups = 32 if (ch>=32 and ch%32==0) else ch
        self.head = nn.Conv2d(input_ch, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks, self.upblocks = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_res_blocks): # Attention is not applied yet
            self.downblocks.append(ResBlock(in_ch=ch, out_ch=ch, dropout=0.1,
                                            num_groups=num_groups, attn=False))
        for _ in range(num_res_blocks+1): # Output of self.head is also concatenated
            self.upblocks.append(ResBlock(in_ch=2*ch, out_ch=ch, dropout=0.1,
                                          num_groups=num_groups, attn=False))
        self.middleblocks = nn.ModuleList([
            ResBlock(ch, ch, dropout=0.1, num_groups=num_groups, attn=True),
            ResBlock(ch, ch, dropout=0.1, num_groups=num_groups, attn=False),
        ])
        self.tail = nn.Sequential(
            nn.GroupNorm(num_groups, ch),
            Swish(),
            nn.Conv2d(ch, input_ch, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x):
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h)
            hs.append(h)
        for layer in self.middleblocks:
            h = layer(h)
        for layer in self.upblocks:
            h = layer(torch.cat([h, hs.pop()], dim=1))
        h = self.tail(h)
        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(ch=32, num_res_blocks=2, dropout=0.1, num_groups=32)
    x = torch.randn(batch_size, 3, 32, 32)
    y = model(x)