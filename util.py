from datetime import datetime
import torch


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

    return name
    
def load_checkpoint(model, opt, name):
    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['opt_state_dict'])

