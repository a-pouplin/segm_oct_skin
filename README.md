## Segmentation with Unet

Unet is a convolutional neural network developed by [Ronnenberg et al, 2015](https://arxiv.org/abs/1505.04597), with the shape of an autoencoder containing multiple skip connections. It was initially designed for segmenting biomedical images.

The code presented in this repository has been designed such that: 
1. the architecture of Unet can be easily modified with different kernel sizes (`--kernel_size`), kernel number (`--kernel_num`), number of layers (`--depth`) 
2. the optimisation part can also be easily modified with various optimisers (`--optim`)
3. the results are all saved in `.npz` that can be processed with the script: `read_experiments.py`. 

However, the last activation layer and loss function have to be modified to handle binary images. 

### Before running any script
- Two external folders are required outside this repository: `data`, that should contain the images saved as `healthy-261-trilabel.pkl`, and `experiments` that will contain all the results.
- All the packages required are listed in requirements.txt
- The data consists of 261 images, of size 2048x682 (reduced to 256x512 for computation purposes) with three labels: epidermis, dermis and background.

### Running and processing an experiment
- `python main.py`, followed by different options. Run Unet on the images. Results are saved as `exp_x`, with `x` the experiment number.
- `python read_experiments.py --nb x`, process the results. `x` being the number of the folder, in `exp_x`.


