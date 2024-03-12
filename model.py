# import json
import torch, math
from torch import nn
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt
from network.unet import UNet
from network.resblk import ResBlock


class DenoisingModule(nn.Module):
    def __init__(self, input_size, input_ch,
                 feat_net_type, feat_net_cfg,
                 alpha, alpha_bar, alpha_bar_prev,
                 large_entropy=True, use_ddim=False):
        super().__init__()
        assert feat_net_type in ['UNet', 'ResBlock']
        if feat_net_type == 'ResBlock':
            self.noise_pred = ResBlock(input_size, input_ch, feat_net_cfg)
        elif feat_net_type == 'UNet': 
            self.noise_pred = UNet(**feat_net_cfg, input_size=input_size, input_ch=input_ch)
        self.use_ddim = use_ddim
        # Coefficients are non-trainable, only relative to timestep T,
        # shared by all channels and all pixels, with broadcast
        if not use_ddim: # Coefficient coef_z for sampling noise is needed
            # When large_entropy is False, lower bound on reverse process entropy, else upper bound
            # Under lower bound case, since alpha_bar_prev for timestep 1 is defined as 1.0, coef_z for that is 0
            coef_z = torch.sqrt(1.0-alpha) if large_entropy else \
                     torch.sqrt((1.0-alpha_bar_prev)*(1.0-alpha)/(1.0-alpha_bar))
            self.register_buffer('coef_z', coef_z)
        # "prev" is defined on DDIM's subsequence of timesteps if use_ddim is True
        # Note that for tau_1 of DDIM, alpha_bar_prev is also defined as 1.0
        coef_input = torch.sqrt(alpha_bar_prev/alpha_bar) if use_ddim else torch.sqrt(1.0/alpha)
        coef_eps = torch.sqrt(1.-alpha_bar_prev)-torch.sqrt(alpha_bar_prev*(1.-alpha_bar)/alpha_bar) \
                   if use_ddim else -(1.0-alpha)/torch.sqrt(alpha*(1-alpha_bar))
        self.register_buffer('coef_input', coef_input)
        self.register_buffer('coef_eps', coef_eps)

    def forward(self, x, z, mode='normal'):
        assert mode in ['normal', 'with_pred_eps', 'pred_eps_only']
        pred_eps = self.noise_pred(x)
        if mode == 'pred_eps_only':
            assert z is None
            return pred_eps
        pred_mean = self.coef_input*x+self.coef_eps*pred_eps
        # DDIM's deterministic sampling
        if self.use_ddim:
            assert z is None
            pred_x_prev = pred_mean
        else: # Normal sampling with sampling noise
            assert z is not None
            pred_x_prev = pred_mean+self.coef_z*z
        if mode == 'with_pred_eps':
            return pred_x_prev, pred_eps
        else: return pred_x_prev


class DenoisingNet(nn.Module):
    def __init__(self, T, net_type, cfg, ratio_cfg, ratio_size, img_size=32, img_ch=3, beta_1=0.0001, beta_T=0.02, tau_S=-1):
        super().__init__()
        self.img_ch = img_ch
        # Note that index 0 is corresspond to math variable of subscript 1
        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # t_series from 1 to T (both end included) with interval t_step
        self.use_ddim = tau_S > 0 and tau_S <= T
        t_step = 1 if not self.use_ddim else (T-1) / (tau_S-1)
        self.t_series = torch.arange(T, 0, -t_step).to(dtype=torch.int32)
        self.t_series = list(self.t_series.numpy()[::-1])
        assert self.t_series[0] == 1 and self.t_series[-1] == T
        t_series_idx = torch.tensor(self.t_series, dtype=torch.long)-1
        # Coefficients for forward process (add noise) 
        self.register_buffer('coef_x_0_forward', torch.sqrt(alphas_bar)[t_series_idx])
        self.register_buffer('coef_noise_forward', torch.sqrt(1.0-alphas_bar)[t_series_idx])
        self.t_to_idx = dict(zip(self.t_series, list(range(len(self.t_series)))))
        self.net = nn.ModuleList()
        # Denoise from timestep T to 0, finally get x_0
        # To get x_0, alpha_bar_0 (i.e. alpha_bar_prev for timestep 1) is needed
        # For both DDPM and DDIM scenario, it is defined as 1.0
        cfg_idx_lut, self.size_lut = [], []
        assert math.isclose(sum(ratio_cfg), 1.0) and len(cfg) == len(ratio_cfg)
        for i, ratio in enumerate(ratio_cfg):
            length = math.ceil(ratio*T)
            length = min(length, T-len(cfg_idx_lut))
            cfg_idx_lut.extend([i]*length)
        assert math.isclose(sum(ratio_size), 1.0) and 2**len(ratio_size) <= img_size
        for i, ratio in enumerate(ratio_size):
            length = math.ceil(ratio*T)
            length = min(length, T-len(self.size_lut))
            self.size_lut.extend([img_size//2**i]*length)
        for idx, t in enumerate(self.t_series):
            alpha, alpha_bar = alphas[t-1], alphas_bar[t-1]
            # index 0 of self.t_series means t=1, which is made sure by assertion before 
            if idx == 0: alpha_bar_prev = torch.tensor(1.0, dtype=alpha_bar.dtype)
            else: alpha_bar_prev = alphas_bar[self.t_series[idx-1]-1]
            layer_cfg_idx = cfg_idx_lut[t-1]
            layer_cfg = cfg[layer_cfg_idx]
            last_layer_size = layer_size if idx != 0 else None
            layer_size = self.size_lut[t-1]
            denoising = DenoisingModule(layer_size, img_ch, net_type, layer_cfg,
                                        alpha, alpha_bar, alpha_bar_prev,
                                        use_ddim=self.use_ddim)
            # Upsample during denoising process (reversed t series)
            upsample = nn.Upsample(last_layer_size, mode='bilinear', align_corners=False) \
                       if last_layer_size and (last_layer_size != layer_size) else None
            layer = nn.ModuleDict({'denoising': denoising, 'upsample': upsample})
            self.net.append(layer)

    # x_0 denotes pure input image
    # Shape of tensor Z is (B, len(t_series), C, H, W) for original DDPM, and None for DDIM sampling
    # If param t is not None and in range [1, T], only noise prediction of timestep t to get x_(t-1) is calculated
    def forward(self, x, Z, t='all'):
        assert len(x.shape) == 4 # (B, C, H, W)
        if t in ['all', 'stacked']:
            num_layer = len(self.net)
            if self.use_ddim: assert Z is None
            else: assert isinstance(Z, list) and len(Z) == num_layer
            if t == 'stacked': x_all, pred_eps_all = [x], []
            for index in range(num_layer-1, -1, -1):
                layer_size = self.size_lut[self.t_series[index]-1]
                assert x.shape[-1] == x.shape[-2] == layer_size
                if not self.use_ddim:
                    z = Z[index]
                    assert len(z.shape) == 4 and \
                           (z.shape[-1] == z.shape[-2] == layer_size)
                else: z = None
                layer = self.net[index]
                if t == 'stacked':
                    x, pred_eps = layer['denoising'](x, z, mode='with_pred_eps')
                    x_all.insert(0, x)
                    pred_eps_all.insert(0, pred_eps)
                    if layer['upsample']: x = layer['upsample'](x)
                else:
                    x = layer['denoising'](x, z, mode='normal')
                    if layer['upsample']: x = layer['upsample'](x)
            if t == 'stacked':
                return x_all, pred_eps_all
            return x
        # Return predicted noise from timestep t to (t-1)
        elif t in self.t_series:
            assert Z is None
            layer_size = self.size_lut[t-1]
            assert len(x.shape) == 4 and \
                   (x.shape[-1] == x.shape[-2] == layer_size)
            layer = self.net[self.t_to_idx[t]]
            pred_eps = layer['denoising'](x, None, mode='pred_eps_only')
            return pred_eps
        return None
    
    # Utilities for training and sampling
    def get_t_series(self): return self.t_series

    # Get single noisy image x_T or noise_t from N(0, I),
    # or all sampling noises Z, or both x_T and Z
    def get_noise(self, n, device, mode='all', t=None):
        assert mode in ['x_T', 'noise_t', 'Z', 'all']
        if mode in ['noise_t', 'x_T', 'all']:
            # Get x_T, t equals T
            if mode in ['x_T', 'all']:
                t = self.t_series[-1]
            assert t is not None
            size_x_t = self.size_lut[t-1]
            x_t = torch.randn(n, self.img_ch, size_x_t, size_x_t).to(device)
            if mode in ['noise_t', 'x_T']:
                return x_t
            x_T = x_t
        if mode in ['Z', 'all']:
            if not self.use_ddim:
                Z = [None] * len(self.t_series)
                for idx, t in enumerate(self.t_series):
                    size_x_t = self.size_lut[t-1]
                    Z[idx] = torch.randn(n, self.img_ch, size_x_t, size_x_t).to(device)
            else: Z = None
            if mode == 'Z': return Z
        return x_T, Z
    
    # Add noise_t to x_0, return x_t
    def add_noise(self, x_0, noise_t, t):
        assert t in self.t_series
        t_idx = self.t_to_idx[t]
        coef_x_0 = self.coef_x_0_forward[t_idx]
        coef_noise = self.coef_noise_forward[t_idx]
        size_x_t = self.size_lut[t-1]
        x_0 = interpolate(x_0, size_x_t, mode='bilinear', align_corners=False)
        return x_0*coef_x_0+noise_t*coef_noise
    
def _img_tensor_to_np(img_tensor):
    img = (img_tensor.clip(-1, 1)+1.0)/2.0
    # Rounding: Add 0.5 -> Clamp
    img = img.mul(255).add_(0.5).clamp_(0, 255)
    img_np = img.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return img_np

if __name__ == '__main__':
    # 1 - Build the network
    device, sample_size = torch.device('cpu'), 5
    # net_type, cfg = 'ResBlock', [{'stage_ch': [32, 128, 128, 256, 128, 3], 'block_in_stage': [1, 2, 2, 2, 2, 1]}]
    net_type, cfg = 'UNet', [{'ch': 128, 'ch_mult': [1, 2, 2, 2], 'attn': [1], 'num_res_blocks': 2, 'dropout': 0.1},
                             {'ch': 64, 'ch_mult': [1, 2], 'attn': [1], 'num_res_blocks': 1, 'dropout': 0.1},
                             {'ch': 32, 'ch_mult': [1, 2], 'attn': [1], 'num_res_blocks': 2, 'dropout': 0.1},
                             {'ch': 32, 'ch_mult': [1], 'attn': [], 'num_res_blocks': 1, 'dropout': 0.1},
                             {'ch': 16, 'ch_mult': [1, 2], 'attn': [], 'num_res_blocks': 1, 'dropout': 0.1}]
    ratio_cfg, ratio_size = [0.1, 0.35, 0.25, 0.15, 0.15], [0.5, 0.2, 0.1, 0.1, 0.1]
    model = DenoisingNet(1000, net_type, cfg, ratio_cfg, ratio_size, tau_S=20).to(device).eval()
    # print(model)

    # state_dict = torch.load('./ckpts/ckpt.pt', map_location=device)['net_model']
    # model.load_state_dict(state_dict)

    # 2 - Dump the network configurations
    # print(json.dumps({'net_type': net_type, 'cfg': cfg, 'ratio_cfg': ratio_cfg, 'ratio_size': ratio_size}, indent=4))

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
    sample_noises = model.get_noise(sample_size, device, 'all')
    with torch.no_grad():
        x_all, pred_eps_all = model(*sample_noises, t='stacked')
    torch.save({'x': x_all, 'pred_eps': pred_eps_all}, 'samples.pth')

    # 7 - Display the sampled image(s)
    num_t_display, samples = 10, torch.load('samples.pth')
    x_all, pred_eps_all = samples['x'], samples['pred_eps']
    assert len(x_all) == len(pred_eps_all)+1
    x_indexes = torch.linspace(0, len(x_all)-1, num_t_display, dtype=torch.int64)
    eps_indexes = (x_indexes-1)[1:]
    x_all = [x_all[i] for i in x_indexes]
    pred_eps_all = [pred_eps_all[i] for i in eps_indexes]
    num_sample = x_all[0].shape[0]
    fig_h, fig_w = num_sample*2, num_t_display
    fig = plt.figure()
    for i in range(num_sample):
        for j in range(num_t_display):
            img_x = _img_tensor_to_np(x_all[j][i, ...])
            fig_idx_x = (i*2)*fig_w+j+1
            fig.add_subplot(fig_h, fig_w, fig_idx_x)
            plt.imshow(img_x)
            if j != 0: # Not the first column
                img_eps = _img_tensor_to_np(pred_eps_all[j-1][i, ...])
                fig_idx_eps = (i*2+1)*fig_w+j+1
                fig.add_subplot(fig_h, fig_w, fig_idx_eps)
                plt.imshow(img_eps)
    for ax in fig.axes:
        ax.set_axis_off()
    plt.savefig('samples.png')
    # plt.show() # plt.show() create a new figure

    # 8 - Readd noise to certain timestep (t_s[-1])
    t_s = model.get_t_series()
    noise_t = model.get_noise(sample_size, device, mode='noise_t', t=t_s[-1])
    x_t = model.add_noise(x_all[0], noise_t, t_s[-1])
    print(x_t.shape)
