MODEL_ARCHITECTURE_ARGS = [
    # Format: [NAME, TYPE/CHOICES, HELP STRING, DEFAULT/REQ]
    ['architecture', str, 'Model architecture', 'REQUIRED'],
    ['in_channels', int, 'Size of each input sample', 'REQUIRED'],
    ['hidden_channels', int, 'Size of each hidden sample', 'REQUIRED'],
    ['num_layers', int, 'Number of message passing layers', 'REQUIRED'],
    ['out_channels', int, 'Number of output channels', None],
    ['dropout', float, 'Dropout probability', 0.],
    ['act', str, 'Activation function', "relu"],
    ['act_first', bool, 'Whether to apply activation before normalization', False],
    ['norm', str, 'Normalization function', None],
    ['jk', str, 'Jumping Knowledge mode', None],
    # Add more arguments as needed
]

def add_args_to_parser(arg_list, parser):
    for arg_name, arg_type, arg_help, arg_default in arg_list:
        has_choices = (type(arg_type) == list) 
        kwargs = {
            'type': type(arg_type[0]) if has_choices else arg_type,
            'help': f"{arg_help} (default: {arg_default})"
        }
        if has_choices: kwargs['choices'] = arg_type
        parser.add_argument(f'--{arg_name}', **kwargs)
    return parser