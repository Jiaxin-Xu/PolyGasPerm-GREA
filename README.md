#  graph machine learning for polymer property predictions

## Data

Data (separated gas csv) are placed at '../data/gas_prop/raw'

## Requirements

This code was developed and tested with Python 3.9.13 and PyTorch 1.10.1
All dependencies are specified in the ```package-list.txt``` file. The packages can be installed by
```
conda create -n myenv --file package-list.txt
```


## Usage

Following are commands to train the graph neural network models.

```

# N2 with cached model
python main.py --add_fp ECFP_MACCS --dataset N2 --out caches 

# O2 without cached model
python main.py --add_fp ECFP_MACCS --dataset O2

# CO2 without cached model
python main.py --add_fp MACCS --dataset CO2

# H2 without cached model
python main.py --add_fp ECFP --dataset H2

# CH4 without cached model
python main.py --add_fp ECFP --dataset CH4


```

