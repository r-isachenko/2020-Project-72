import argparse
import json

from data_loader import get_data_loader
from models import get_models, get_params, predict
from loss import get_losses, compute_losses
from optimizer import get_optimizer

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

    model_params = get_params(models)
    optimizer = get_optimizer(params['optimizer'], model_params)

    for step, (x, y) in enumerate(train_loader):
        loss_values = compute_losses(losses, models, x, y)
        optimizer.zero_grad()

        loss_values['total_loss'].backward()
        optimizer.step()

        if step % params['summary']['train_frequency'] == 0:
            pass

        if step % params['summary']['save_frequency'] == 0:
            pass

        break


if __name__ == '__main__':
    args = parse_args()
    main(args)