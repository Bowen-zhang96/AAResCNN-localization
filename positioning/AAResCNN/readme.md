## Instruction

## Trained models
The trained models are stored in dictionary bestmodels/ for testing

## Training
The main file is positioning.py. Please follow the steps below for training.

1. Change the dataset path in line 26 and 180 according to where the dataset is stored in your computer.
2. Change the antenna layout type in line 22, 23, 194, 203 to the considered layout. The options include 'ULA', 'DIS', 'URA'
3. Change the number of training samples in line 203, 235, 258-260, 264. The options include '1000', '5000', '10000'
4. Run positioning.py


## Testing
The main file is positioning.py. Please follow the steps below for testing.

1. Change the dataset path in line 26 and 180 according to where the dataset is stored in your computer.
2. Change the antenna layout type in line 22, 23, 194, 203 to the considered layout. The options include 'ULA', 'DIS', 'URA'
3. Comment out code from line 235 to 260 to skip training
4. Change the name of the restored models in line 264. Ensure the restored model has the right antenna type, the number of training sample or suffix
5. Run positioning.py
