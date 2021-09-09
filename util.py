import torch

def save_checkpoint(model, opt, epoch):
    name = (type(model).__name__ 
        + '-Checkpoint-e{}-'.format(epoch) 
        + datetime.now().strftime('%m-%d@%H-%M'))
    
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
        }, 
        '{}.pt'.format(name)
    )
    
def load_checkpoint(model, opt, name):
    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['opt_state_dict'])
