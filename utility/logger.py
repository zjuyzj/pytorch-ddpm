import os
from tensorboardX import SummaryWriter

from utility.misc import save_img_tensor, make_img_grid
from utility.misc import append_to_json


class TensorboardLogger:
    def __init__(self, log_dir, purge_step=None):
        self.step, self.log_dir = 0, log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir, purge_step=purge_step)
        self.sample_dir = os.path.join(log_dir, 'sample')
        os.makedirs(self.sample_dir, exist_ok=True)

    def set_step(self, step=None):
        if step is not None:
            self.step += 1
        else: self.step = step

    def write_img(self, name, img_tensor, postfix='', save_png=False):
        if len(img_tensor.shape) != 3:
            img_tensor = make_img_grid(img_tensor)
        if len(postfix) != 0: name = f"{name}-{postfix}"
        self.writer.add_image(name, img_tensor, self.step)
        self.writer.flush()
        if not save_png: return
        img_filename = str(self.step)
        if len(postfix) != 0: img_filename += f"_{postfix}"
        img_path = os.path.join(self.sample_dir, f'{img_filename}.png')
        save_img_tensor(img_tensor, img_path)

    def write_data(self, name, data_scalar, postfix=''):
        if len(postfix) != 0: name = f"{name}-{postfix}"
        self.writer.add_scalar(name, data_scalar, self.step)
        self.writer.flush()

    def write_data_dict(self, data_scalar_dict, postfix=''):
        for name, data_scalar in data_scalar_dict.items():
            if len(postfix) != 0: name = f"{name}-{postfix}"
            self.writer.add_scalar(name, data_scalar, self.step)
        self.writer.flush()

    def write_config_bak_file(self, kwargs: dict):
        config_bak_path = os.path.join(self.log_dir, "flagfile.txt")
        f = open(config_bak_path, 'w')
        f.write(kwargs['argv_string'])
        f.close()

    def write_eval_file(self, data_scalar_dict):
        eval_path = os.path.join(self.log_dir, "eval.txt")
        data_scalar_dict['step'] = self.step
        append_to_json(eval_path, data_scalar_dict)

    def __del__(self):
        self.writer.close()