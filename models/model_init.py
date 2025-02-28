from models.checkpoints_dict import get_latest_checkpoint

def model_init(params):
    if params['method'] == 'UNet':
        from models.unet_basic import UNet as model
    elif params['method'] == 'UNetContrast':
        from models.unet_contrastive import UNet as model 
    else:
        raise NotImplementedError
    
    if not (params['ckpt_in'] is None):
        print(''.join(['-']*80))
        print('Loading checkpoint from: ', params['ckpt_in'])
        ckpt_path = get_latest_checkpoint(params['ckpt_in'])

        # Correctly use load_from_checkpoint on the class
        my_model = model.load_from_checkpoint(ckpt_path, **params)
        print(''.join(['-']*80))
    else:
        # Initialize the model if there is no checkpoint
        my_model = model(**params)

    return my_model

    