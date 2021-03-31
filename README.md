# BTP Sem 1 Mid Sem 

image2spike.py is python code that reads images from MNIST handwritten digits dataset and converts it into spike signals. Theses spike signals are then fed into a Piece-wise Linear(PWL) voltage source in LtSpice.

## How to use

* Step 1:
    Download MNIST Dataset from [this link][https://www.kaggle.com/c/digit-recognizer/data].

* Step 2:
    Download python code and put "train.csv" in same directory.

* Step 3:
    If you read the python code currently it is wroking for image_name = "5". It can be changed for different images.

* Step 4:
    Run the python code. An output text file named "voltage_for_{image_name}" is produced.

* Step 5:
    Make a voltage source in Ltspice (PWL type) and give the pathname of the text file generated in previous step.

* Step 6:
    Simlulate the voltage source using ".tran 150m" command and observe the waveforms
