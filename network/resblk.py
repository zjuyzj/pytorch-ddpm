from torch import nn
from torch.nn import init

# Simple ResNet block without downsampling and upsampling
class ResBlock(nn.Module):
    def __init__(self, input_size, input_ch, feat_net_cfg):
        super().__init__()
        self.feat_net = nn.ModuleList()
        stage_ch = feat_net_cfg['stage_ch']
        block_in_stage = feat_net_cfg['block_in_stage']
        assert len(stage_ch) >= 2
        if stage_ch[-1]: stage_ch[-1] = input_ch
        assert stage_ch[-1] == input_ch
        assert len(stage_ch) == len(block_in_stage)
        for stage_idx in range(len(stage_ch)):
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
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)
                # module.bias is always None yet
                if module.bias is not None:
                    init.zeros_(module.bias)
        for block_idx in range(len(self.feat_net)//3):
            init.xavier_uniform_(self.feat_net[3*block_idx][-2].weight, gain=1e-5)

    def forward(self, x):
        for block_idx in range(len(self.feat_net)//3):
            conv, sc, act = self.feat_net[3*block_idx:3*block_idx+3]
            x_sc = sc(x)
            x = conv(x) + x_sc
            x = act(x)
        z = x # Predicted noise
        return z
