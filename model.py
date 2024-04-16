import json
import torch, math
from torch import nn
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt
from network.unet import UNet
from network.resnet import ResNet
from network.unet_t import UNet_T


class DenoisingModule(nn.Module): # The smallest unit for layer-by-layer training
    def __init__(self, input_size, input_ch, feat_net_type, feat_net_cfg,
                 alpha, alpha_bar, alpha_bar_prev, large_entropy=True, use_ddim=False,
                 resize_fn={'func_x': None, 'func_eps': None, 'func_x_prev': None}):
        super().__init__()
        # Mode A: 'func_x_prev' | Mode B: 'func_x'+'func_eps'
        assert set(resize_fn.keys()) == {'func_x', 'func_eps', 'func_x_prev'}
        assert feat_net_type in ['UNet', 'ResNet', None]
        if feat_net_type == 'ResNet':
            self.noise_pred = ResNet(input_size, input_ch, feat_net_cfg)
        elif feat_net_type == 'UNet': 
            self.noise_pred = UNet(**feat_net_cfg, input_size=input_size, input_ch=input_ch)
        elif feat_net_type is None: self.noise_pred = None
        self.use_ddim, self.resize_fn = use_ddim, resize_fn
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

    def forward(self, x, z, mode='normal', predictor={'noise_pred': None, 't': None}):
        assert mode in ['normal', 'with_pred_eps', 'pred_eps_only', 'pred_eps_only_resized']
        if mode in ['pred_eps_only', 'pred_eps_only_resized']:
            assert z is None
        if self.noise_pred is None:
            assert predictor['noise_pred'] is not None
        if mode == 'pred_eps_only':
            return self.noise_pred(x) if self.noise_pred \
                   else predictor['noise_pred'](x, predictor['t'])
        input_x = x # Backup for the original x_t
        if self.resize_fn['func_x']:
            x = self.resize_fn['func_x'](x)
        pred_eps = self.noise_pred(x) if self.noise_pred \
                   else predictor['noise_pred'](x, predictor['t'])
        if self.resize_fn['func_eps']:
            pred_eps = self.resize_fn['func_eps'](pred_eps)
        if mode == 'pred_eps_only_resized': return pred_eps
        pred_mean = self.coef_input*input_x+self.coef_eps*pred_eps
        # DDIM's deterministic sampling
        if self.use_ddim:
            assert z is None
            pred_x_prev = pred_mean
        else: # Normal sampling with sampling noise
            pred_x_prev = pred_mean+self.coef_z*z
        if self.resize_fn['func_x_prev']:
            pred_x_prev = self.resize_fn['func_x_prev'](pred_x_prev)
        if mode == 'with_pred_eps':
            return pred_x_prev, pred_eps
        else: return pred_x_prev


class DenoisingNet(nn.Module):
    def __init__(self, T, net_type, cfg, ratio_cfg,
                 ratio_size, div_factor_size, resize_policy='x_t', loss_policy='raw', 
                 img_size=32, img_ch=3, beta_1=0.0001, beta_T=0.02, tau_S=-1,
                 multi_step_diffusion=False):
        super().__init__()
        assert resize_policy in ['x_t', 'eps']
        assert loss_policy in ['raw', 'resized']
        # Always calc loss on epsilon under current framework
        if resize_policy == 'x_t': assert loss_policy == 'raw'
        self.resize_policy, self.loss_policy, = resize_policy, loss_policy
        # Get and add Gaussian noise progressively rather than at once
        self.multi_step_diffusion = multi_step_diffusion
        self.img_size, self.img_ch = img_size, img_ch
        # Note that index 0 corresponds to math variable of subscript 1
        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # t_series from 1 to T (both end included) with interval t_step
        self.use_ddim = tau_S > 0 and tau_S <= T
        t_step = 1 if not self.use_ddim else (T-1) / (tau_S-1)
        self.t_series = list(torch.arange(1, T+1, t_step).to(dtype=torch.int32).numpy())
        assert self.t_series[0] == 1 and self.t_series[-1] == T
        t_series_idx = torch.tensor(self.t_series, dtype=torch.long)-1
        # Coefficients for forward process (add noise) 
        if not multi_step_diffusion:
            self.register_buffer('coef_x_0_forward', torch.sqrt(alphas_bar)[t_series_idx])
            self.register_buffer('coef_noise_forward', torch.sqrt(1.0-alphas_bar)[t_series_idx])
        else: # In multi-step forward diffusion, timestep that not sampled during generation are also needed
            self.register_buffer('coef_x_prev_forward', torch.sqrt(alphas))
            self.register_buffer('coef_noise_forward', betas)
        self.t_to_idx = dict(zip(self.t_series, list(range(len(self.t_series)))))
        self.net = nn.ModuleList()
        # Denoise from timestep T to 0, finally get x_0
        # To get x_0, alpha_bar_0 (i.e. alpha_bar_prev for timestep 1) is needed
        # For both DDPM and DDIM scenario, it is defined as 1.0
        cfg_idx_lut, self.size_lut = [], []
        assert math.isclose(sum(ratio_cfg), 1.0) and len(cfg) == len(ratio_cfg)
        if net_type == 'UNet_T': # Only timesteps used are embedded and trained
            assert len(cfg) == 1 # Multi-conf and resizing are not supported 
            assert len(ratio_size) == 1 and div_factor_size[0] == 1
            self.global_noise_pred = UNet_T(T=len(self.t_series), **cfg[0]) # Shared noise predictor
            net_type = None # Avoid DenoisingModule to generate internal noise predictor
        else: self.global_noise_pred = None
        for i, ratio in enumerate(ratio_cfg):
            length = math.ceil(ratio*T)
            length = min(length, T-len(cfg_idx_lut))
            cfg_idx_lut.extend([i]*length)
        assert len(ratio_size) == len(div_factor_size)
        assert math.isclose(sum(ratio_size), 1.0)
        assert img_size//max(div_factor_size) >= 2
        for i, ratio in enumerate(ratio_size):
            length = math.ceil(ratio*T)
            length = min(length, T-len(self.size_lut))
            self.size_lut.extend([img_size//div_factor_size[i]]*length)
        for idx, t in enumerate(self.t_series):
            alpha, alpha_bar = alphas[t-1], alphas_bar[t-1]
            # index 0 of self.t_series means t=1, which is made sure by assertion before 
            if idx == 0: alpha_bar_prev = torch.tensor(1.0, dtype=alpha_bar.dtype)
            else: alpha_bar_prev = alphas_bar[self.t_series[idx-1]-1]
            layer_cfg = cfg[cfg_idx_lut[t-1]]
            if resize_policy == 'x_t': # To keep the size of x_t continuous
                last_layer_size = layer_size if idx != 0 else None
            layer_size = self.size_lut[t-1]
            if resize_policy == 'x_t':
                upsample = nn.Upsample(last_layer_size, mode='bilinear', align_corners=False) \
                           if last_layer_size and (last_layer_size != layer_size) else None
                resize_fn = {'func_x': None, 'func_eps': None, 'func_x_prev': upsample}
            # Denoising on img_size, and noise prediction on layer_size (input_size)
            elif resize_policy == 'eps' and layer_size != img_size:
                if loss_policy == 'resized': # SUCCEED
                    downsample = nn.PixelUnshuffle(img_size//layer_size)
                    upsample = nn.PixelShuffle(img_size//layer_size)
                else: # FAIL no matter which resize_policy is
                    # i.e. optimization problem (loss value about 0.77)
                    # under 'resized' while abnormal sampling under 'raw'
                    downsample = nn.Upsample(layer_size, mode='bilinear', align_corners=False)
                    upsample = nn.Upsample(img_size, mode='bilinear', align_corners=False)
                resize_fn = {'func_x': downsample, 'func_eps': upsample, 'func_x_prev': None}
            else: resize_fn = {'func_x': None, 'func_eps': None, 'func_x_prev': None}
            # When PixelShuffle and PixelUnshuffle is used, layer_ch for noise predictor's I/O is img_ch*(scale_factor^2)
            layer_ch = img_ch*(img_size//layer_size)**2 if resize_policy == 'eps' and loss_policy == 'resized' else img_ch
            layer = DenoisingModule(layer_size, layer_ch, net_type, layer_cfg,
                                    alpha, alpha_bar, alpha_bar_prev,
                                    use_ddim=self.use_ddim, resize_fn=resize_fn)
            self.net.append(layer)

    # x_0 denotes pure input image
    # Shape of tensor Z is (B, len(t_series), C, H, W) for original DDPM, and None for DDIM sampling
    # If param t is not None and in range [1, T], only noise prediction of timestep t to get x_(t-1) is calculated
    def forward(self, x, Z, mode='all', t=None):
        if mode in ['all', 'stacked']:
            num_layer = len(self.net)
            if not self.use_ddim:
                assert isinstance(Z, list) and len(Z) == num_layer
            else: assert Z is None
            if mode == 'stacked': x_all, pred_eps_all = [x], []
            if t is not None: assert t in self.t_series
            index_layer_start = num_layer-1 if t is None else self.t_to_idx[t]
            for index in range(index_layer_start, -1, -1):
                z = Z[index] if not self.use_ddim else None
                layer = self.net[index]
                if self.global_noise_pred is not None:
                    timesteps = torch.ones((x.shape[0]), dtype=torch.long, device=x.device)*index
                    predictor = {'noise_pred': self.global_noise_pred, 't': timesteps}
                else: predictor={'noise_pred': None}
                if mode == 'stacked':
                    x, pred_eps = layer(x, z, mode='with_pred_eps', predictor=predictor)
                    x_all.insert(0, x)
                    pred_eps_all.insert(0, pred_eps)
                else:
                    x = layer(x, z, mode='normal', predictor=predictor)
            if mode == 'stacked':
                return x_all, pred_eps_all
            else: return x
        # Return predicted noise from timestep t to (t-1)
        elif mode == 'single':
            assert t in self.t_series and Z is None
            layer = self.net[self.t_to_idx[t]]
            if self.global_noise_pred is not None:
                timesteps = torch.ones((x.shape[0]), dtype=torch.long, device=x.device)*self.t_to_idx[t]
                predictor = {'noise_pred': self.global_noise_pred, 't': timesteps}
            else: predictor={'noise_pred': None}
            # Return the original output of noise predictor to calc loss
            if self.loss_policy == 'raw':
                pred_eps = layer(x, None, mode='pred_eps_only', predictor=predictor)
            # Return the resized pred_eps to calc loss, not available when 
            # resize_mode is 'x_t' because loss is calculated on epsilon
            elif self.loss_policy == 'resized':
                pred_eps = layer(x, None, mode='pred_eps_only_resized', predictor=predictor)
            return pred_eps
        return None
    
    # Utilities for training and sampling
    def get_t_series(self): return self.t_series

    # Get single noisy image x_T or noise_t from N(0, I),
    # or all sampling noises Z, or both x_T and Z
    # noise_t is only used for training
    def get_noise(self, n, device, mode='all', t=None):
        # x_t+raw: noise_t - layer_size, x_T and z - layer_size -> blurred
        # eps+raw: noise_t - layer_size, x_T and z - img_size -> problematic sampling
        # eps+resized: noise_t - img_size, x_T and z - img_size -> relative OK now
        assert mode in ['x_T', 'noise_t', 'Z', 'all']
        if mode == 'noise_t':
            assert t in self.t_series
            if self.multi_step_diffusion: noise_t_collected = list()
            # t noises is needed to get x_t from x_0
            t_all = list(range(1, t+1)) if self.multi_step_diffusion else [t]
            for t in t_all: # Resample the noises for different timesteps
                if self.loss_policy == 'raw':
                    layer_size = self.size_lut[t-1] 
                    noise_t_shape = (self.img_ch, layer_size, layer_size)
                elif self.loss_policy == 'resized':
                    noise_t_shape = (self.img_ch, self.img_size, self.img_size)
                noise_t = torch.randn(n, *noise_t_shape).to(device)
                if not self.multi_step_diffusion: return noise_t
                else: noise_t_collected.append(noise_t)
            return noise_t_collected
        assert t is None
        if mode in ['x_T', 'all']:
            x_T_size = self.img_size if self.resize_policy == 'eps' \
                       else self.size_lut[self.t_series[-1]-1]
            x_T_shape = (self.img_ch, x_T_size, x_T_size)
            x_T = torch.randn(n, *x_T_shape).to(device)
            if mode == 'x_T': return x_T
        if mode in ['Z', 'all']:
            if not self.use_ddim:
                Z = [None] * len(self.t_series)
                for idx, t in enumerate(self.t_series):
                    z_size = self.img_size if self.resize_policy == 'eps' \
                             else self.size_lut[t-1]
                    z_shape = (self.img_ch, z_size, z_size)
                    Z[idx] = torch.randn(n, *z_shape).to(device)
            else: Z = None
            if mode == 'Z': return Z
        return x_T, Z
    
    # Add noise to x_0, return x_t
    # 'noise_t' is single noise if added at once, otherwise collected noise
    # Always do interpolation on x_0, and never on noise
    def add_noise(self, x_0, noise_t, t):
        assert t in self.t_series
        # Add noises with a progressive scheme
        if self.multi_step_diffusion:
            assert len(noise_t) == t
            x_prev = x_0 # Add noise from x_0
            for t_idx in range(t):
                noise_t_single = noise_t[t_idx]
                coef_x_prev = self.coef_x_prev_forward[t_idx]
                coef_noise = self.coef_noise_forward[t_idx]
                x_prev = interpolate(x_prev, noise_t_single.shape[-1], mode='bilinear', align_corners=False)
                x_prev = x_prev*coef_x_prev+noise_t_single*coef_noise
            return x_prev
        else: # Add single noise to x_0 at once
            t_idx = self.t_to_idx[t]
            coef_x_0 = self.coef_x_0_forward[t_idx]
            coef_noise = self.coef_noise_forward[t_idx]
            x_0 = interpolate(x_0, noise_t.shape[-1], mode='bilinear', align_corners=False)
            return x_0*coef_x_0+noise_t*coef_noise
        
    def get_multi_ground_truth(self, x_0, noise_t):
        if not self.multi_step_diffusion:
            return None
        # 'ground_truth' is used to calculate loss
        #  with t-1 final and intermediate and result 
        ground_truth, x_prev = [x_0], x_0
        for t in range(1, len(noise_t)):
            noise_t_single = noise_t[t-1]
            coef_x_prev = self.coef_x_prev_forward[t-1]
            coef_noise = self.coef_noise_forward[t-1]
            x_prev = x_prev*coef_x_prev+noise_t_single*coef_noise
            if t in self.t_series: ground_truth.append(x_prev)
        return ground_truth
    
    # ID for excluded layer start from 1
    def load_checkpoint(self, state_dict, layer_excluded=[]):
        if self.global_noise_pred is not None:
            assert len(layer_excluded) == 0
            self.load_state_dict(state_dict, strict=True)
            return
        for layer in layer_excluded:
            assert layer > 0 and layer <= len(self.net)
        # Convert layer ID to layer index
        layer_excluded, keys_to_del = [id-1 for id in layer_excluded], []
        for key in state_dict.keys():
            fields = key.split('.')
            if len(fields) < 2: continue
            layer_id = int(fields[1])
            if layer_id in layer_excluded:
                keys_to_del.append(key)
        # Convert the state_dict for compatibility reason, e.g. load
        # layer-by-layer pretrained checkpoint for end2end fine-tuning
        keys_to_del.append('coef_x_0_forward')
        keys_to_del.append('coef_noise_forward')
        for key in keys_to_del:
            del state_dict[key]
        result = self.load_state_dict(state_dict, strict=False)
        assert len(result.unexpected_keys) == 0
        # Use newly constructed version of these keys
        keys_allowed_missing = ['coef_x_prev_forward', 'coef_noise_forward']
        for missing_key in result.missing_keys:
            if missing_key in keys_allowed_missing: continue
            # For deleted layers, key missing is also allowed
            fields = missing_key.split('.')
            assert len(fields) >= 2 and fields[0] == 'net'
            assert int(fields[1]) in layer_excluded
        return


def _img_tensor_to_np(img_tensor):
    img = (img_tensor.clip(-1, 1)+1.0)/2.0
    # Rounding: Add 0.5 -> Clamp
    img = img.mul(255).add_(0.5).clamp_(0, 255)
    img_np = img.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return img_np

if __name__ == '__main__':
    # 1 - Load network configurations and build the network
    device, sample_size = torch.device('cpu'), 5
    cfg_path = './config/net/unet_flex_small.json'
    with open(cfg_path, 'r') as f:
        cfg_dict = json.loads(f.read())
    model = DenoisingNet(1000, **cfg_dict, tau_S=100).to(device).eval()
    # print(model)

    # 2 - Dump the network configurations
    # print(json.dumps(cfg_dict, indent=4))

    # 3 - Network loading and saving
    # torch.save(model.state_dict(), 'test.pt')
    # model.load_state_dict(torch.load('test.pt'))

    # 4 - Show learnable parameters
    # for i, (name, p) in enumerate(model.named_parameters()):
    #     print(i, name, p.requires_grad)

    # 5 - Print the number of parameters
    print('Training:', model.training)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#PARAMS:", num_params)

    # 6 - Sample from the model
    sample_noises = model.get_noise(sample_size, device, 'all')
    with torch.no_grad():
        x_all, pred_eps_all = model(*sample_noises, mode='stacked')
    torch.save({'x': x_all, 'pred_eps': pred_eps_all}, 'samples.pth')

    # 7 - Display the sampled image(s)
    num_t_displayed, samples = 10, torch.load('samples.pth')
    x_all, pred_eps_all = samples['x'], samples['pred_eps']
    assert len(x_all) == len(pred_eps_all)+1
    x_indexes = torch.linspace(0, len(x_all)-1, num_t_displayed, dtype=torch.int64)
    eps_indexes = (x_indexes-1)[1:]
    x_all = [x_all[i] for i in x_indexes]
    pred_eps_all = [pred_eps_all[i] for i in eps_indexes]
    num_sample = x_all[0].shape[0]
    fig_h, fig_w = num_sample*2, num_t_displayed
    fig = plt.figure()
    for i in range(num_sample):
        for j in range(num_t_displayed):
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

    # 8 - Readd noise to certain timestep (t_s[-1] a.k.a. T)
    t_s = model.get_t_series()
    noise_T = model.get_noise(sample_size, device, mode='noise_t', t=t_s[-1])
    x_T = model.add_noise(x_all[0], noise_T, t_s[-1])
    print(x_T.shape)
    pred_eps = model(x_T, None, mode='single', t=t_s[-1])
    print(pred_eps.shape)
