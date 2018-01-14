# Implementation of Deep-Forecast using PyTorch

* [Deep Forecast: Deep Learning-based Spatio-Temporal Forecasting](https://arxiv.org/pdf/1707.08110.pdf)
* Adapted from [original implementation](https://github.com/amirstar/Deep-Forecast)

## Setup

* Clone this repository : `git clone https://github.com/Wizaron/deep-forecast-pytorch.git`
* Download and install [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://conda.io/miniconda.html)
* Go to the "reseg-pytorch/code/pytorch" : `cd reseg-pytorch/code/pytorch`
* Create environment : `conda env create -f conda_environment.yml`
* Activate environment : `source activate deep-forecast-pytorch`

## Code Structure

* "data" : Stores data and scripts to prepare dataset for training.
* "lib" : Stores miscellaneous scripts for training and testing.
  ** "arch.py" : Defines network architecture
  ** "model.py" : Defines model (Minibatching mechanism, optimization, criterion, fit, predict, etc.)
  ** "prediction.py" : Metrics and plots to evaluate the performance of the trained model
  ** "data.py" : Creates training, validation and testings datasets
  ** "loader.py" : Creates Dataset loader for PyTorch
* "train.py" : Main training script.
* "settings.py" : Defines hyper-parameters of the model.

## Data

* Data is downloaded from [IEM](https://mesonet.agron.iastate.edu/request/download.phtml)
* Download data and save it under "data/raw"
* To prepare dataset, run the scripts in "data/scripts"

## Training and Testing

* Train and test : `python train.py --data [PATH OF PREPARED DATASET]`
* For more info : `python train.py --help`

### train.py

* It saves models and logs under "models"
* At the end of the training, it saves predictions under "outputs"
