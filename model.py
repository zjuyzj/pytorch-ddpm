# import json
import torch, math
from torch import nn
from torch.nn import init
import matplotlib.pyplot as plt
from UNet import UNet


'''
# Simple ResNet block without downsampling and upsampling
class NoisePredModule(nn.Module):
    def __init__(self, input_size, input_ch, feat_net_cfg):
        super().__init__()
        self.feat_net = nn.ModuleList()
        stage_ch = feat_net_cfg['stage_ch']
        block_in_stage = feat_net_cfg['block_in_stage']
        assert len(stage_ch) >= 2 and stage_ch[-1] == input_ch
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
'''


class DenoisingModule(nn.Module):
    def __init__(self, input_size, input_ch, feat_net_cfg,
                 alpha, alpha_bar, alpha_bar_prev,
                 large_entropy=True, use_ddim=False):
        super().__init__()
        # self.noise_pred = NoisePredModule(input_size, input_ch, feat_net_cfg)
        self.noise_pred = UNet(**feat_net_cfg) # DEBUG, comment this to save memory
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

    # def forward(self, x, z, unet_timestep, mode='normal'): # DEBUG
    def forward(self, x, z, mode='normal'):
        assert mode in ['normal', 'with_pred_eps', 'pred_eps_only']
        # unet_timesteps = torch.ones((x.shape[0]), dtype=torch.long) # DEBUG
        # unet_timesteps = (unet_timesteps*unet_timestep).to(device) # DEBUG
        # pred_eps = unet_t(x, unet_timesteps).detach() # DEBUG
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
    def __init__(self, T, cfg, cfg_ratio, input_size=32, input_ch=3, beta_1=0.0001, beta_T=0.02, tau_S=-1):
        super().__init__()
        self.img_size = (input_ch, input_size, input_size)
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
        self.register_buffer('coef_input_forward', torch.sqrt(alphas_bar)[t_series_idx])
        self.register_buffer('coef_z_forward', torch.sqrt(1.0-alphas_bar)[t_series_idx])
        self.t_to_idx = dict(zip(self.t_series, list(range(len(self.t_series)))))
        self.net = nn.ModuleList()
        # Denoise from timestep T to 0, finally get x_0
        # To get x_0, alpha_bar_0 (i.e. alpha_bar_prev for timestep 1) is needed
        # For both DDPM and DDIM scenario, it is defined as 1.0
        cfg_idx_lut = []
        assert sum(cfg_ratio) <= 1.0 and len(cfg) == len(cfg_ratio)
        for i, ratio in enumerate(cfg_ratio):
            length = math.ceil(ratio*T)
            length = min(length, T-len(cfg_idx_lut))
            cfg_idx_lut.extend([i]*length)
        for idx, t in enumerate(self.t_series):
            alpha, alpha_bar = alphas[t-1], alphas_bar[t-1]
            # index 0 of self.t_series means t=1, which is made sure by assertion before 
            if idx == 0: alpha_bar_prev = torch.tensor(1.0, dtype=alpha_bar.dtype)
            else: alpha_bar_prev = alphas_bar[self.t_series[idx-1]-1]
            layer_cfg_idx = cfg_idx_lut[t-1]
            layer_cfg = cfg[layer_cfg_idx]
            layer = DenoisingModule(input_size, input_ch, layer_cfg,
                                    alpha, alpha_bar, alpha_bar_prev,
                                    use_ddim=self.use_ddim)
            self.net.append(layer)

    # Shape of tensor Z is (B, len(t_series)+1, C, H, W) for original DDPM, and (B, 1, C, H, W) for DDIM sampling
    # [~, len(t_series), ~, ~, ~] means input Gaussian noise for denoising, and [~, i, ~, ~, ~],
    # with i in range [0, len(t_series)-1], is the sampling noise to get x_i.
    # If param t is not None and in range [1, T], size of the second dim of Z before must be 1 or not exist,
    # only noise prediction of timestep t, which is used to to get x_(t-1), is calculated.
    def forward(self, Z, t=None):
        # x_0 denotes pure input image
        if t is None or t == 'all':
            if len(Z.shape) == 4:
                Z = Z.unsqueeze(dim=1)
            assert len(Z.shape) == 5
            x = Z[:, -1, ...]
            num_layer = len(self.net)
            if t == 'all': x_all, pred_eps_all = [x], []
            for index in range(num_layer-1, -1, -1):
                layer = self.net[index]
                z = Z[:, index, ...] if not self.use_ddim else None
                if t == 'all': # Collect intermediate x and eps in reference
                    # unet_timestep = list(self.t_to_idx.keys())[list(self.t_to_idx.values()).index(index)] - 1 # DEBUG
                    # x, pred_eps = layer(x, z, unet_timestep, mode='with_pred_eps') # DEBUG
                    x, pred_eps = layer(x, z, mode='with_pred_eps')
                    x_all.insert(0, x)
                    pred_eps_all.insert(0, pred_eps)
                else:
                    x = layer(x, z, mode='normal')
            if t == 'all':
                x_all = torch.stack(x_all, dim=0)
                pred_eps_all = torch.stack(pred_eps_all, dim=0)
                return x_all, pred_eps_all
            else: return x
        # Return predicted noise from timestep t to (t-1)
        elif t in self.t_series:
            x = Z if len(Z.shape) == 4 else Z.squeeze()
            layer = self.net[self.t_to_idx[t]]
            pred_eps = layer(x, None, mode='pred_eps_only')
            return pred_eps
        return None # Illegal timestep t
    
    # Utilities for training and sampling
    def get_t_series(self): return self.t_series

    # Get single noisy image x_T from N(0, I), or all noises
    # including x_T that sampling needed (x_T given or unknown)
    def get_noise(self, n, device, mode='all', x_T=None):
        if mode == 'single' or (mode == 'all' and self.use_ddim):
            assert x_T is None
            return torch.randn(n, *self.img_size).to(device)
        elif mode == 'all' and not self.use_ddim:
            num_z = len(self.t_series) # Number of sampling noises
            if x_T is None:
                return torch.randn(n, num_z+1, *self.img_size).to(device)
            noises = torch.randn(n, num_z, *self.img_size).to(device)
            x_T = x_T.to(device)
            if len(x_T.shape) == 4:
                x_T = x_T.unsqueeze(dim=1)
            assert len(x_T.shape) == 5
            return torch.cat([noises, x_T], dim=1)
        return None
    
    # Add noise z to x (noise level t), return x_t(x_0) 
    def add_noise(self, x, z, t):
        assert t in self.t_series
        t_idx = self.t_to_idx[t]
        coef_x = self.coef_input_forward[t_idx]
        coef_z = self.coef_z_forward[t_idx]
        return x*coef_x+z*coef_z
    
def _img_tensor_to_np(img_tensor):
    img = (img_tensor.clip(-1, 1)+1.0)/2.0
    # Rounding: Add 0.5 -> Clamp
    img = img.mul(255).add_(0.5).clamp_(0, 255)
    img_np = img.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return img_np

if __name__ == '__main__':
    # 1 - Build the network
    device, sample_size = torch.device('cpu'), 5
    # cfg = [{'stage_ch': [32, 128, 128, 256, 128, 3],
    #         'block_in_stage': [1, 2, 2, 2, 2, 1]}]
    cfg = [{'ch': 128, 'ch_mult': [1, 2, 2, 2], 'attn': [1], 'num_res_blocks': 2, 'dropout': 0.1}]
    cfg_ratio = [1.0]
    model = DenoisingNet(1000, cfg, cfg_ratio, tau_S=20).to(device).eval()
    # print(model)

    # state_dict = torch.load('./ckpts/ckpt.pt', map_location=device)['net_model']
    # model.load_state_dict(state_dict)

    # DEBUG: Test U-Net for original DDPM, T=1000, w or w/o DDIM sampling
    # from UNet_T import UNet_T
    # unet_t = UNet_T(1000, 128, [1, 2, 2, 2], [1], 2, 0.1).to(device).eval()
    # state_dict = torch.load('./ckpts/backups/ckpt.pt', map_location=device)['ema_model']
    # unet_t.load_state_dict(state_dict)

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
    Z = model.get_noise(sample_size, device, 'all')
    with torch.no_grad():
        x_all, pred_eps_all = model(Z, t='all')
    torch.save({'x': x_all, 'pred_eps': pred_eps_all}, 'samples.pth')

    # 7 - Display the sampled image(s)
    num_t_display, samples = 10, torch.load('samples.pth')
    x_all, pred_eps_all = samples['x'], samples['pred_eps']
    assert x_all.shape[0] == (pred_eps_all.shape[0]+1)
    x_indexes = torch.linspace(0, x_all.shape[0]-1, num_t_display, dtype=torch.int64)
    eps_indexes = (x_indexes-1)[1:]
    x_all, pred_eps_all = x_all[x_indexes], pred_eps_all[eps_indexes]
    num_sample = x_all.shape[1]
    fig_h, fig_w = num_sample*2, num_t_display
    fig = plt.figure()
    for i in range(num_sample):
        for j in range(num_t_display):
            img_x = _img_tensor_to_np(x_all[j, i, ...])
            fig_idx_x = (i*2)*fig_w+j+1
            fig.add_subplot(fig_h, fig_w, fig_idx_x)
            plt.imshow(img_x)
            if j != 0: # Not the first column
                img_eps = _img_tensor_to_np(pred_eps_all[j-1, i, ...])
                fig_idx_eps = (i*2+1)*fig_w+j+1
                fig.add_subplot(fig_h, fig_w, fig_idx_eps)
                plt.imshow(img_eps)
    for ax in fig.axes:
        ax.set_axis_off()
    plt.savefig('samples.png')
    # plt.show() # plt.show() create a new figure

    # 8 - Readd noise to certain timestep (t_s[-1])
    # t_s = model.get_t_series()
    # noise = model.get_noise(sample_size, device, 'single')
    # x_t = model.add_noise(x_all[0], noise, t_s[-1])
    # print(x_t.shape)
