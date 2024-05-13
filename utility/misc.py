import json, torch, warnings
from torchvision.utils import make_grid, save_image


def make_img_grid(img_batch_tensor: torch.Tensor):
    return make_grid(img_batch_tensor)


def save_img_tensor(img_tensor: torch.Tensor, img_path: str):
    if len(img_tensor.shape) != 3:
        assert len(img_tensor.shape) == 4
        img_tensor = make_img_grid(img_tensor)
    save_image(img_tensor, img_path)


def load_from_json(json_path: str):
    with open(json_path) as f:
        data = json.loads(f.read())
    return data


def append_to_json(json_path:str, data, indent=None):
    f = open(json_path, 'a')
    f.write(json.dumps(data, indent=indent) + "\n")
    f.close()


def set_warning_level():
    warnings.simplefilter("ignore", category=FutureWarning)
    # warnings.simplefilter("ignore", category=UserWarning)