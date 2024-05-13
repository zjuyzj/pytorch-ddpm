import torch, math
from torch import nn
from .network.unet import UNet
from .network.resnet import ResNet


class DenoisingModule(nn.Module): # The smallest unit for layer-by-layer training
    def __init__(self, input_size, input_ch, feat_net_type, feat_net_cfg, alpha_bar, alpha_bar_prev):
        super().__init__()
        assert feat_net_type in ['UNet', 'ResNet', None]
        if feat_net_type == 'ResNet':
            self.noise_pred = ResNet(input_size, input_ch, feat_net_cfg)
        elif feat_net_type == 'UNet': 
            self.noise_pred = UNet(**feat_net_cfg, input_size=input_size, input_ch=input_ch)
        elif feat_net_type is None: self.noise_pred = None
        coef_input = torch.sqrt(alpha_bar_prev/alpha_bar)
        coef_eps = torch.sqrt(1.-alpha_bar_prev)-torch.sqrt(alpha_bar_prev*(1.-alpha_bar)/alpha_bar)
        self.register_buffer('coef_input', coef_input)
        self.register_buffer('coef_eps', coef_eps)

    def forward(self, x, x_gt=None):
        pred_eps = self.noise_pred(x)
        input_x = x if x_gt is None else x_gt 
        return self.coef_input*input_x+self.coef_eps*pred_eps


class DenoisingNet(nn.Module):
    def __init__(self, device, T, tau, net_type, cfg, ratio_cfg, img_size=32, img_ch=3, beta_1=0.0001, beta_T=0.02):
        super().__init__()
        self.img_shape, self.device = (img_ch, img_size, img_size), device
        alphas = 1.0-torch.linspace(beta_1, beta_T, T, dtype=torch.float64)
        alphas_bar = torch.cumprod(alphas, dim=0)
        timesteps = torch.arange(1, T+1, (T-1)/(tau-1)).to(dtype=torch.long)
        self.register_buffer('alphas_bar', alphas_bar[timesteps-1])
        self.register_buffer('coef_x_0_forward', torch.sqrt(alphas_bar)[timesteps-1])
        self.register_buffer('coef_noise_forward', torch.sqrt(1.0-alphas_bar)[timesteps-1])
        self.net, cfg_idx_lut = nn.ModuleList(), []
        assert math.isclose(sum(ratio_cfg), 1.0) and len(cfg) == len(ratio_cfg)
        for i, ratio in enumerate(ratio_cfg):
            length = math.ceil(ratio*T)
            length = min(length, T-len(cfg_idx_lut))
            cfg_idx_lut.extend([i]*length)
        for idx, t in enumerate(timesteps):
            alpha_bar = alphas_bar[t-1]
            alpha_bar_prev = torch.tensor(1.0, dtype=alpha_bar.dtype) \
                             if idx == 0 else alphas_bar[timesteps[idx-1]-1]
            layer_cfg = cfg[cfg_idx_lut[t-1]]
            layer = DenoisingModule(img_size, img_ch, net_type, layer_cfg, alpha_bar, alpha_bar_prev)
            self.net.append(layer)

    def forward(self, x, x_gt_all=None):
        pred_x_all = []
        for index in range(len(self.net)-1, -1, -1):
            layer = self.net[index]
            if x_gt_all is not None and index != len(self.net)-1:
                x_gt = x_gt_all[:, index+1, ...] 
            else: x_gt = None
            x = layer(x, x_gt)
            pred_x_all.insert(0, x)
        return torch.stack(pred_x_all, dim=1)

    def get_alphas_bar(self):
        return self.alphas_bar
    
    def get_device(self):
        return self.device

    def get_noise(self, n):
        return torch.randn(n, *self.img_shape).to(self.device)

    def add_noise(self, x_0, noise):       
        coef_x_0 = self.coef_x_0_forward[-1]
        coef_noise = self.coef_noise_forward[-1]
        return x_0*coef_x_0+noise*coef_noise

    def get_multi_ground_truth(self, x_0, noise):
        ground_truth, num_gt = [x_0], self.coef_x_0_forward.shape[0]-1
        for idx in range(num_gt):
            coef_x_0 = self.coef_x_0_forward[idx]
            coef_noise = self.coef_noise_forward[idx]
            gt_single = x_0*coef_x_0+noise*coef_noise
            ground_truth.append(gt_single)
        return torch.stack(ground_truth, dim=1)

    # It only works with model checkpoint which is not wrapped with DDP
    def load_checkpoint(self, state_dict, layer_excluded=[]):
        for layer in layer_excluded:
            assert layer > 0 and layer <= len(self.net)
        layer_excluded, keys_to_del = [id-1 for id in layer_excluded], []
        for key in state_dict.keys():
            fields = key.split('.')
            if len(fields) < 2: continue
            layer_id = int(fields[1])
            if layer_id in layer_excluded:
                keys_to_del.append(key)
        for key in keys_to_del:
            del state_dict[key]
        result = self.load_state_dict(state_dict, strict=False)
        assert len(result.unexpected_keys) == 0
        for missing_key in result.missing_keys:
            fields = missing_key.split('.')
            assert len(fields) >= 2 and fields[0] == 'net'
            assert int(fields[1]) in layer_excluded
    
    def print_model_size(self):
        model_size = 0
        for param in self.parameters():
            if not param.requires_grad: continue
            model_size += param.data.nelement()
        model_size /= 1024*1024
        print('Model Parameters: %.2f M'%model_size)