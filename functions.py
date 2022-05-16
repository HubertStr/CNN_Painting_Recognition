def freeze_up_to_layer(mod, block_number = 7):
    '''
    Input: 
        block_number: float; number of the given block not included
    '''
    ct = 0
    for child in mod.children():
        ct += 1
        if ct < block_number:
            for param in child.parameters():
                param.requires_grad = False
    return mod