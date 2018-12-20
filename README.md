# Shape Classifier

Uses [Google QuickDraw](https://github.com/1991viet/QuickDraw) dataset to recognize between :

- circle
- hourglass
- square
- star
- triangle

Used to use dataset generated by [Shape Drawer](https://shape-drawer.netlify.com/).

## Installation

+ `pipenv install`

## Usage

+ `cd src/`
+ `python download_data`
+ `python train.py`

## Structure

+ `Graph/` contains the recording of each training
+ `models/` contains all the models trained
+ `src/` contains the code of the model
