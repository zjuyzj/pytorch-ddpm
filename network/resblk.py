import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


class AttnBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(ch)
        self.proj_q = nn.Conv2d(ch, ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(ch, ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(ch, ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(ch, ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.batch_norm(x)
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


# Simple ResNet block without downsampling and upsampling
class ResBlock(nn.Module):
    def __init__(self, input_size, input_ch, feat_net_cfg):
        super().__init__()
        self.feat_net = nn.ModuleList()
        stage_ch = feat_net_cfg['stage_ch']
        block_in_stage = feat_net_cfg['block_in_stage']
        assert len(stage_ch) >= 1 and len(stage_ch) == len(block_in_stage)
        for stage_idx in range(len(stage_ch)):
            for block_idx in range(block_in_stage[stage_idx]):
                in_ch = input_ch if stage_idx == 0 and block_idx == 0 \
                        else stage_ch[stage_idx-1] if block_idx == 0 \
                        else stage_ch[stage_idx]
                out_ch = stage_ch[stage_idx]
                layers = nn.BatchNorm2d(in_ch), \
                         nn.ReLU(inplace=True), \
                         nn.Conv2d(in_ch, out_ch, 3, 1, 1), \
                         nn.BatchNorm2d(out_ch), \
                         nn.ReLU(inplace=True), \
                         nn.Conv2d(out_ch, out_ch, 3, 1, 1)
                self.feat_net.append(nn.Sequential(*layers))
                shortcut = nn.Identity() if in_ch == out_ch else \
                           nn.Conv2d(in_ch, out_ch, 1, 1, 0)
                self.feat_net.append(shortcut)
                self.feat_net.append(AttnBlock(out_ch))
        tail = nn.BatchNorm2d(stage_ch[-1]), \
               nn.ReLU(inplace=True), \
               nn.Conv2d(stage_ch[-1], input_ch, 3, stride=1, padding=1)
        self.tail = nn.Sequential(*tail)
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
        for block_idx in range(len(self.feat_net)//3):
            init.xavier_uniform_(self.feat_net[3*block_idx][-1].weight, gain=1e-5)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)

    def forward(self, x):
        for block_idx in range(len(self.feat_net)//3):
            conv, sc, attn = self.feat_net[3*block_idx:3*block_idx+3]
            x_sc = sc(x)
            x = conv(x) + x_sc
            x = attn(x)
        noise_pred = self.tail(x)
        return noise_pred
