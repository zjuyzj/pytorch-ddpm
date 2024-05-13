from torch import nn
from torch.nn import init


class ResNet(nn.Module):
    def __init__(self, input_size, input_ch, feat_net_cfg):
        super().__init__()
        self.feat_net = nn.ModuleList()
        stage_ch = feat_net_cfg['stage_ch']
        block_in_stage = feat_net_cfg['block_in_stage']
        assert len(stage_ch) >= 1 and len(stage_ch) == len(block_in_stage)
        for stage_idx in range(len(stage_ch)):
            # ResNet block without downsampling and upsampling
            for block_idx in range(block_in_stage[stage_idx]):
                in_ch = input_ch if stage_idx == 0 and block_idx == 0 \
                        else stage_ch[stage_idx-1] if block_idx == 0 \
                        else stage_ch[stage_idx]
                out_ch = stage_ch[stage_idx]
                layers = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False), \
                         nn.BatchNorm2d(out_ch), \
                         nn.ReLU(inplace=True), \
                         nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), \
                         nn.BatchNorm2d(out_ch)
                self.feat_net.append(nn.Sequential(*layers))
                shortcut = nn.Identity() if in_ch == out_ch else \
                           nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
                                         nn.BatchNorm2d(out_ch))
                self.feat_net.append(shortcut)
                self.feat_net.append(nn.ReLU(inplace=True))
        self.tail = nn.Conv2d(stage_ch[-1], input_ch, 3, stride=1, padding=1)
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if not isinstance(module, nn.Conv2d): continue
            init.xavier_uniform_(module.weight)
            if module.bias is None: continue
            init.zeros_(module.bias)
        # Set the network output to zero (gray image) at the beginning
        init.xavier_uniform_(self.tail.weight, gain=1e-5)
        return

    def forward(self, x):
        for block_idx in range(len(self.feat_net)//3):
            conv, sc, act = self.feat_net[3*block_idx:3*block_idx+3]
            x_sc = sc(x)
            x = conv(x) + x_sc
            x = act(x)
        noise_pred = self.tail(x)
        return noise_pred
