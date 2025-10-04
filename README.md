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
python src/run.py --config src/config/xyz.cfg
```
## Citation
If you use our codes in your research, please cite:
```
@INPROCEEDINGS{11169623,
  author={Li, Qibo and Zhao, Yuqi and Ma, Yutao},
  booktitle={2025 IEEE International Conference on Web Services (ICWS)}, 
  title={TDIC: Time-Aware Disentanglement of Interest and Conformity in Mobile App Recommendations}, 
  year={2025},
  pages={1-10},
  doi={10.1109/ICWS67624.2025.00011}}

```
