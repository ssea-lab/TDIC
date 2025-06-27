# TDIC
The goal of this project is to model user interest and conformity in a time-aware manner. TDIC disentangles interest and conformity using item popularity.


## Environment
* Python = 3.10.14
* Pytorch = 2.1.0
* numpy = 1.26.4
* tqdm = 4.64.4
### Train & Test
Train and evaluate the model with the following commands.
You can also add command parameters to specify dataset/epoch, or set neg-sampling rate.

```shell
# set parameters in config.py 
# --dataset: (str) dataset name
# --epochs: (int) epoch numbers
# --neg_sample_rate: (int) negative sample numbers
python main.py
```
