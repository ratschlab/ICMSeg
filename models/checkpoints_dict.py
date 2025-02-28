import os

def get_latest_checkpoint(ckpt_file):
    path = ckpt_file
    print('Ckpt path was explicitly provided:\t', ckpt_file)

    if '.ckpt' in path:
        print('Path is not a folder and is already a checkpoint!')
        return path
    
    print(f"Searching for checkpoints in directory: {path}")
    checkpoints = [f for f in os.listdir(path) if 'ckpt' in f]
    print(checkpoints)
    latest_ckpt = None
    ts = 0
    for ckpt_file in checkpoints:
        filename = os.path.join(path, ckpt_file)
        filename_ts = os.path.getmtime(filename)
        if filename_ts > ts:
            latest_ckpt = filename
            ts = filename_ts
            
    print('Latest checkpoint:', latest_ckpt)
    quit()
    return latest_ckpt