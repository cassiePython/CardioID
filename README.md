# Introduction

This is a pytorch implementation for the paper CadioID: learning to identification from electrocardiogram data. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231220310766). 


# Usage

```
# Train the model - we give default parameters in options/train_options.py
python train_physionet.py

# test the model - we give default parameters in options/test_options.py
python test_physionet.py
```

# Requirements

python 3.7.5, pytorch 1.2.0


If you find the idea useful or use this code in your own work, please cite our paper as
```
@article{hong2020cardioid,
  title={CardioID: Learning to Identification from Electrocardiogram Data},
  author={Hong, Shenda and Wang, Can and Fu, Zhaoji},
  journal={Neurocomputing},
  year={2020},
  publisher={Elsevier}
}
```