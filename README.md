# Single-Cell Image Super-Resolution with GAN

ETH Data Science Lab Biology Team

## Requirements
    1. PyTorch 0.4.1
    2. tqdm

## Usage

### Example usage
    
    python main.py --model SRGAN --lr 0.001 --upscale_factor 8 --batch_size 100 --epochs 100 --n_save 2 --gamma 0.001 --theta 0.01
        
The above command trains a GAN-based model, with upscaling factor 8.
        
### Details
see models/train.py for commandline options.

### Examples

Go to https://zhaopku.github.io/sr.html
