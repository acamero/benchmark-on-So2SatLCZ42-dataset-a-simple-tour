# LCZ classification from so2satLCZ42 dataset

## basics and requirements

keras (tensorflow backend)

data:
https://arxiv.org/abs/1912.12171

http://doi.org/10.14459/2018MP1454690


## Folder Structure
  ```
  benchmark-on-So2SatLCZ42-dataset-a-simple-tour/
  ├── dataLoader.py - loading data from h5 files
  ├── evaluation.py - evaluation of the trained models
  ├── lr.py - learning rate schedule
  ├── model.py - architecture of sen2LCZ_drop
  ├── plotModel.py - plot the models
  ├── requirements.txt - python environment requirements
  ├── resnet.py - resnet model implementation
  ├── train.py - main file for training (path to data needs to be set)
  │
  │
  ├── img2map/ - predict using the trained models from s2 data
  │   ├── img2lczMap_oneCity.py - read s2 data and predict and save the results in geotiff
  │   ├── img2mapC4Lcz.py - functions for predictions
  │   └── ...
  │
  └── modelFig/ - figure of the model structure
  │   ├── sen2LCZ_drop.png - sen2LCZ_drop model plot
  │
  ├── results/ - results folder
  │   ├── 00017_22007_Lagos.tif - test example (city of Lagos) prediction
  │   ├── Lagos_sen2LCZ.png - test example (city of Lagos) overlayed on OSM
  │   ├── pre-trained models
  │   └── ...
  ```
## Usage

Clone the repository and install all the dependencies listed in `requirements.txt`. Note: the code was tested using Python 3.6.9.

### img2map: predict the LCZ using the pre-trained models

1. Download the data from [image data for test](https://drive.google.com/drive/u/1/folders/1y-lFSuUeY3barjKJVG1TTwh39RqlUzn6)
2. Run the code, `CUDA_VISIBLE_DEVICES=0 python img2lczMap_oneCity.py <path_to_model> <path_to_data>`
, e.g., `CUDA_VISIBLE_DEVICES=0 python img2lczMap_oneCity.py "../results/sen2LCZ_32_weights.best.hdf5" "testData/00017_22007_Lagos"`
3. Check the predicted values (e.g., `results/Lagos_sen2LCZ.png`)

### train

To train a model from scratch, 
1. download the [data](http://doi.org/10.14459/2018MP1454690)
2. set the data path (in train.py) 
3. Run the code, `CUDA_VISIBLE_DEVICES=0 python train.py`


