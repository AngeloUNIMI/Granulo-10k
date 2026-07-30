def pc_warmup_alpha(epoch, warmup_epochs):
    if epoch >= warmup_epochs:
        return 1.0
    return epoch / warmup_epochs