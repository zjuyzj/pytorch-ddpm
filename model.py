# import json
import torch, math
from torch import nn
from torch.nn import init
from PIL import Image
from UNet import UNet

'''
class NoisePredModule(nn.Module):
    def __init__(self, input_size, input_ch, feat_net_cfg):
        super().__init__()
        # The feature extraction network is based on the shallow ResNet without bottleneck, and no stem.
        # i.e. two 3x3 conv layers in one block with a shortcut connection, any number of block in one stage,
        # downsample and double the number of channels only at the first layer of the first block in one stage
        self.feat_net, multi_scale_ch = nn.ModuleList(), []
        # The final output of the sub-network must bigger than 2x2
        init_ch, block_in_stage = feat_net_cfg['init_ch'], feat_net_cfg['block_in_stage']
        mid_ch, mid_size = feat_net_cfg['mid_ch'], feat_net_cfg['mid_size']
        assert input_size >= pow(2, len(block_in_stage)+1)
        for stage_idx, block_num in enumerate(block_in_stage):
            is_first_stage = stage_idx == 0
            for block_idx in range(block_num):
                is_first_block = block_idx == 0
                do_downsampling = is_first_block
                if not (is_first_stage and is_first_block):
                    in_ch = out_ch
                    out_ch = 2*in_ch if do_downsampling else in_ch
                else: in_ch, out_ch = input_ch, init_ch
                multi_scale_ch.append(out_ch)
                first_layer_stride = 2 if do_downsampling else 1
                layers = nn.Conv2d(in_ch, out_ch, 3, first_layer_stride, 1, bias=False), \
                         nn.BatchNorm2d(out_ch), \
                         nn.ReLU(inplace=True), \
                         nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), \
                         nn.BatchNorm2d(out_ch)
                self.feat_net.append(nn.Sequential(*layers))
                shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, 2, 0, bias=False),
                                         nn.BatchNorm2d(out_ch)) \
                           if do_downsampling else nn.Identity()
                self.feat_net.append(shortcut)
                self.feat_net.append(nn.ReLU(inplace=True))
        self.multi_scale_idx = [sum(block_in_stage[0:i+1])-1 for i in range(len(block_in_stage))]
        multi_scale_ch = [multi_scale_ch[idx] for idx in self.multi_scale_idx]
        self.pre_concat = nn.ModuleList()
        # Construct multi-scale feature volume from CNN pyramid
        for in_ch in multi_scale_ch:
            layer = nn.Conv2d(in_ch, mid_ch, 1, bias=False), \
                    nn.BatchNorm2d(mid_ch), \
                    nn.ReLU(inplace=True), \
                    nn.Upsample(mid_size, mode='bilinear', align_corners=False)
            self.pre_concat.append(nn.Sequential(*layer))
        # Generate the image of predicted noise from feature volume
        pyramid_ch = len(self.pre_concat)*mid_ch
        self.head = nn.Sequential(nn.Conv2d(pyramid_ch, pyramid_ch, 3, 1, 1, bias=True),
                                  nn.Conv2d(pyramid_ch, input_ch, 1, 1, 0, bias=True),
                                  nn.Upsample(input_size, mode='bilinear', align_corners=False))
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x):
        multi_scale_out = []
        for block_idx in range(len(self.feat_net)//3):
            conv, sc, act = self.feat_net[3*block_idx:3*block_idx+3]
            x_sc = sc(x)
            x = conv(x) + x_sc
            x = act(x)
            if block_idx in self.multi_scale_idx:
                multi_scale_out.append(x)
        for idx in range(len(multi_scale_out)):
            seq = self.pre_concat[idx]
            multi_scale_out[idx] = seq(multi_scale_out[idx])
        pyramid = torch.cat(multi_scale_out, dim=1)
        z = self.head(pyramid)
        return z
'''


# Simple ResNet block without downsampling and upsampling
class NoisePredModule(nn.Module):
    def __init__(self, input_size, input_ch, feat_net_cfg):
        super().__init__()
        self.feat_net = nn.ModuleList()
        stage_ch_fact = feat_net_cfg['stage_ch_fact']
        block_in_stage = feat_net_cfg['block_in_stage']
        assert len(stage_ch_fact) >= 2 and stage_ch_fact[-1] == 1
        assert len(stage_ch_fact) == len(block_in_stage)
        for stage_idx in range(len(stage_ch_fact)):
            for block_idx in range(block_in_stage[stage_idx]):
                in_ch = input_ch if stage_idx == 0 and block_idx == 0 \
                        else input_ch*stage_ch_fact[stage_idx-1] if block_idx == 0 \
                        else input_ch*stage_ch_fact[stage_idx]
                out_ch = input_ch*stage_ch_fact[stage_idx]
                layers = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False), \
                        nn.BatchNorm2d(out_ch), \
                        nn.SiLU(inplace=True), \
                        nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), \
                        nn.BatchNorm2d(out_ch)
                self.feat_net.append(nn.Sequential(*layers))
                shortcut = nn.Identity() if in_ch == out_ch else \
                        nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(out_ch))
                self.feat_net.append(shortcut)
                self.feat_net.append(nn.SiLU(inplace=True))
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)
                # module.bias is always None yet
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x):
        for block_idx in range(len(self.feat_net)//3):
            conv, sc, act = self.feat_net[3*block_idx:3*block_idx+3]
            x_sc = sc(x)
            x = conv(x) + x_sc
            x = act(x)
        z = x # Predicted noise
        return z


class DenoisingModule(nn.Module):
    def __init__(self, input_size, input_ch, feat_net_cfg, alpha, alpha_bar, alpha_bar_prev, large_entropy=True):
        super().__init__()
        # self.noise_pred = NoisePredModule(input_size, input_ch, feat_net_cfg)
        self.noise_pred = UNet(**feat_net_cfg)
        # Coefficients are non-trainable, only relative to timestep T,
        # shared by all channels and all pixels, with broadcast
        self.register_buffer('coef_input', torch.sqrt(1.0/alpha))
        self.register_buffer('coef_eps', -(1.0-alpha)/torch.sqrt(alpha*(1-alpha_bar)))
        if not large_entropy: # Option A: Lower bound on reverse process entropy
            self.register_buffer('coef_z', torch.sqrt((1.0-alpha_bar_prev)*(1.0-alpha)/(1.0-alpha_bar)))
        else: # Option B: Upper bound on reverse process entropy
            self.register_buffer('coef_z', torch.sqrt(1.0-alpha))

    # Debug: Test U-Net for original DDPM
    # def forward(self, x, z, t):
    def forward(self, x, z):
        # Debug: Test U-Net for original DDPM
        # pred_eps = unet(x, t).detach()
        pred_eps = self.noise_pred(x)
        pred_mean = self.coef_input*x+self.coef_eps*pred_eps
        return pred_mean+self.coef_z*z


class DenoisingNet(nn.Module):
    def __init__(self, T, cfg, cfg_ratio, input_size=32, input_ch=3, beta_1=0.0001, beta_T=0.02):
        super().__init__()
        self.T, self.img_size = T, (input_ch, input_size, input_size)
        # Note: index 0 is corresspond to math variable of subscript 1
        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # Coefficients for forward process (add noise) 
        self.register_buffer('coef_input_forward', torch.sqrt(alphas_bar))
        self.register_buffer('coef_z_forward', torch.sqrt(1.0-alphas_bar))
        self.net = nn.ModuleList()
        # Denoise from timestep T to 1 (total T-1 layers), finally get x_1
        # To get result x_0 (timestep 0), alpha_bar_0 is needed to calculate 
        # coef_z (timestep 1) in DenoisingModule(), but it is not defined here.
        # If needed, alpha_bar_0 = 1.0, while coef_z is the same with timestep 2.
        cfg_idx_lut = []
        assert sum(cfg_ratio) <= 1.0 and len(cfg) == len(cfg_ratio)
        for i, ratio in enumerate(cfg_ratio):
            length = math.ceil(ratio*(T-1))
            length = min(length, (T-1)-len(cfg_idx_lut))
            cfg_idx_lut.extend([i]*length)
        for t in range(1, T):
            alpha, alpha_bar= alphas[t], alphas_bar[t]
            alpha_bar_prev = alphas_bar[t-1]
            layer_cfg_idx = cfg_idx_lut[t-1]
            layer_cfg = cfg[layer_cfg_idx]
            layer = DenoisingModule(input_size, input_ch, layer_cfg, alpha, alpha_bar, alpha_bar_prev)
            self.net.append(layer)

    # Shape of tensor Z is (B, T, C, H, W)
    # [~, T-1, ~, ~, ~] means input Gaussian noise for denoising, and [~, i, ~, ~, ~],
    # with i in range [0, T-2], is the noise for sampling to get x_(i+1).
    # If param t is not None and in range [2, T], size of the second dim of Z must be 1 or not exist,
    # only noise prediction of timestep t, which is used to to get x_(t-1), is calculated.
    def forward(self, Z, t=None):
        # x_0 denotes pure input image 
        if t is not None: # input is x_t
            assert 2 <= t <= self.T
            x = Z if len(Z.shape) == 4 else Z.squeeze()
            noise_pred, param_in_graph = self.net[t-2].noise_pred, []
            for k, p in noise_pred.named_parameters():
                if not p.requires_grad: continue
                param_in_graph.append(f'net.{t-2}.noise_pred.{k}')
            noise = noise_pred(x)
            return noise, param_in_graph
        else:
            x = Z[:, -1, ...]
            num_layer = len(self.net)
            for index in range(num_layer-1, -1, -1):
                layer = self.net[index]
                z = Z[:, index, ...]
                # Debug: Test U-Net for original DDPM
                # timesteps = torch.ones((x.shape[0]), dtype=torch.long)
                # timesteps = timesteps*(index+1).to(device)
                # x = layer(x, z, timesteps)
                x = layer(x, z)
            return x
    
    # Utilities for training and sampling

    # Get single noisy image x_T from N(0, I), or all noises
    # including x_T that sampling needed (x_T given or unknown)
    def get_noise(self, n, device, mode='all', x_T=None):
        if mode == 'single':
            return torch.randn(n, *self.img_size).to(device)
        elif mode == 'all':
            if x_T is not None:
                noises = torch.randn(n, self.T-1, *self.img_size).to(device)
                x_T = x_T.to(device)
                if len(x_T.shape) == 4:
                    x_T = x_T.unsqueeze(dim=1)
                assert len(x_T.shape) == 5
                return torch.cat([noises, x_T], dim=1)
            else: return torch.randn(n, self.T, *self.img_size).to(device)
        return None
    
    # Add noise z to x (noise level t), return x_t(x_0) 
    def add_noise(self, x, z, t):
        assert 2 <= t <= self.T
        coef_x = self.coef_input_forward[t-1]
        coef_z = self.coef_z_forward[t-1]
        return x*coef_x+z*coef_z

if __name__ == '__main__':
    # 1 - Build the network
    device = torch.device('cpu')
    # cfg = [{'init_ch': 32, 'block_in_stage': [3, 4, 6, 3], 'mid_ch': 16, 'mid_size': 16},
    #        {'init_ch': 32, 'block_in_stage': [2, 2, 2, 2], 'mid_ch': 16, 'mid_size': 16},
    #        {'init_ch': 32, 'block_in_stage': [2, 2, 2], 'mid_ch': 16, 'mid_size': 16}]
    # cfg_ratio = [0.1, 0.3, 0.6]
    # cfg = [{'stage_ch_fact': [1, 4, 8, 16, 64, 32, 16, 8, 4, 1],
    #         'block_in_stage': [1, 2, 4, 4, 2, 2, 4, 4, 2, 1]}]
    cfg = [{'ch': 32, 'ch_mult': [1, 2], 'attn': [1], 'num_res_blocks': 2, 'dropout': 0.1}]
    cfg_ratio = [1.0]
    model = DenoisingNet(100, cfg, cfg_ratio).to(device).eval()
    # print(model)

    # Debug: Test U-Net for original DDPM, T=1000
    # unet = UNet(1000, 128, [1, 2, 2, 2], [1], 2, 0.1).to(device).eval()
    # state_dict = torch.load('./ckpts/backups/ckpt.pt', map_location=device)['ema_model']
    # unet.load_state_dict(state_dict)

    # 2 - Dump the network configurations
    # print(json.dumps({'cfg': cfg, 'cfg_ratio': cfg_ratio}, indent=4))

    # 3 - Network loading and saving
    # torch.save(model.state_dict(), 'test.pt')
    # model.load_state_dict(torch.load('test.pt'))

    # 4.1 - Show learnable parameters (Method A)
    # state_dict, cnt = model.state_dict(keep_vars=True), 0
    # for key in state_dict.keys():
    #     requires_grad = state_dict[key].requires_grad
    #     if not requires_grad:
    #         print('[x]', key)
    #         continue
    #     print(cnt, key)
    #     cnt += 1

    # 4.2 - Show learnable parameters (Method B)
    # for i, (name, p) in enumerate(model.named_parameters()):
    #     print(i, name, p.requires_grad)

    # 5 - Print the number of parameters
    print('Training:', model.training)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#PARAMS:", num_params)

    # 6 - Sample from the model
    Z = model.get_noise(5, device, 'all')
    res = model(Z)
    print(res.shape)

    # 7 - Save the sampled image(s)
    res = (res.clip(-1, 1)+1.0)/2.0
    for i in range(res.shape[0]):
        # Rounding: Add 0.5 -> Clamp
        np_img = res[i, ...].mul(255).add_(0.5).clamp_(0, 255) \
                .permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        Image.fromarray(np_img).save(f'sample_{i}.jpg')

    # 8 - Readd noise to certain timestep
    # x_t = model.add_noise(res, model.get_noise(5, device, 'single'), 100)
    # print(x_t.shape)
