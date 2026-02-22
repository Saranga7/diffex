from templates import *

if __name__ == '__main__':
    # train the autoenc moodel
    gpus = [0, 1]
    conf = bbc_autoenc()
    conf.batch_size = 64 
    conf.accum_batches = 4
    conf.name = 'bbc_021'
    train(conf, gpus=gpus)


    