import argparse
import json

from data_loader import get_data_loader
from models import get_models, predict
from loss import get_losses, compute_losses

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params-path', required=True, type=argparse.FileType('r'),
                        help='Path to a json parameters')
    args = parser.parse_args()

    return args


def read_params(fid):
    return json.loads(fid.read())


def main(args):
    params = read_params(args.params_path)

    train_loader = get_data_loader(params['data_type'], train=True, batch_size=params['batch_size'])
    valid_loader = get_data_loader(params['data_type'], train=False, batch_size=params['batch_size'])

    models = get_models(params['models'])
    losses = get_losses(params['losses'])

    for x, y in train_loader:
        loss_values = compute_losses(losses, models, x, y)
        print(loss_values)
        break


if __name__ == '__main__':
    args = parse_args()
    main(args)