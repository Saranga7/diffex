from dataclasses import dataclass


@dataclass
class Config:
    GPU_INDEX = 2
    SEED = 7
    BATCH_SIZE = 256
    EPOCHS = 10
    LR = 3e-4
    IMG_SIZE = 128

