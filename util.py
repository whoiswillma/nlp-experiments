import os
import sys
from datetime import datetime
import torch
import logging
from tqdm import tqdm
import gc


# PyTorch Processing Unit
def get_ptpu():
    result = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    print(f'PTPU: {result}')
    return result

PTPU = get_ptpu()


def mytqdm(*args, **kwargs):
    return tqdm(*args, ncols=70, leave=False, **kwargs)


def init_logging():
    datetime_str = get_datetime_str()
    log_filename = f'Log_{os.path.basename(sys.argv[0])}-{datetime_str}.log'
    logging.basicConfig(
        format='%(levelname)s:%(asctime)s %(message)s',
        filename=log_filename,
        level=logging.DEBUG
    )
    print(f'Logging to {log_filename}')
    os.system(f'open {log_filename}')


def pytorch_set_num_threads(num_threads):
    logging.info(f'Setting num threads to {num_threads}')
    torch.set_num_threads(num_threads)


def get_datetime_str():
    return datetime.now().strftime('%m-%d@%H-%M')


def save_checkpoint(model, opt, epoch, datetime_str=None):
    if not datetime_str:
        datetime_str = get_datetime_str()

    name = (type(model).__name__ 
        + '-Checkpoint-e{}-'.format(epoch) 
        + datetime_str)
    
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
        }, 
        '{}.pt'.format(name)
    )

    logging.info(f'Saved checkpoint {name}')

    return name
    
def load_checkpoint(model, opt, name):
    checkpoint = torch.load(name)

    if model:
        model.load_state_dict(checkpoint['model_state_dict'])

    if opt:
        opt.load_state_dict(checkpoint['opt_state_dict'])

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

