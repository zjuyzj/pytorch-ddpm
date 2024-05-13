from absl import flags

from .misc import load_from_json

def register_cmd_argument(json_path: str):
    type_str_to_fn = {'str': 'flags.DEFINE_string', 'bool': 'flags.DEFINE_boolean', 
                      'int': 'flags.DEFINE_integer', 'float': 'flags.DEFINE_float',
                      'list': 'flags.DEFINE_spaceseplist'}
    param_all = load_from_json(json_path)
    for params in param_all.values():
        for param in params:
            type_str = str(type(param['default'])).split('\'')[1]
            if type_str == 'list':
                param['default'] = ' '.join(list(map(str, param['default'])))
            eval(type_str_to_fn[type_str])(**param)


def parse_cmd_argument():
    kwargs = flags.FLAGS.flag_values_dict()
    kwargs['argv_string'] = flags.FLAGS.flags_into_string()
    return kwargs