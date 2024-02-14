# Parser configs
import arg_parser_config
from argparse import ArgumentParser

parser = ArgumentParser()
parser = arg_parser_config.add_args_to_parser(arg_parser_config.MODEL_ARCHITECTURE_ARGS, parser)

def main(args):
    model = make_and_restore_model(arch=args.arch, dataset=args.dataset, resume_path=args.resume)


if __name__ == "__main__":
    args = parser.parse_args()
    toolbox = GNNRobustness(args)

    # Choose or define GNN architecture
    toolbox.choose_available_gnn_architectures()