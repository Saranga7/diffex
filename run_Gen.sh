#!/bin/bash

MODEL_PATH="outputs/run/train_direction_models/diffae_k10/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha0.1_BS128/2026-02-22_23-45"

# Run the Python script with the specified arguments
python gen.py \
    --config-path="conf" \
    --config-name=gen \
    checkpoint="$MODEL_PATH/best_model.pt" \
    n_samples=5 \
    alphas="[-15,-10,-5,5,10,15]" \
    iterative=False \
    image_size=128 \
    n_dirs=[0,1,2,3,4] \




# [-5,-3,-1,1,3,5]
# [-7,-5,-3,3,5,7]
# [-15,-10,-5,5,10,15]
