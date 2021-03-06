# BTP Sem 1 Mid Sem Work

image2spike.py is python code that reads images from MNIST handwritten digits dataset and converts it into spike signals. These spike signals are then fed into a Piecewise Linear(PWL) voltage source in LtSpice.

## Requirements

* Python 3.6.9
* NumPy
* Matplotlib
* OpenCV
* Pandas
* LtSpice

Please install above tools and/or libraries for running the code and visualization of waveforms.

## How to use

* Step 1:
    Download MNIST Dataset from [this link](https://www.kaggle.com/c/digit-recognizer/data) .

* Step 2:
    Download python code and put "train.csv" in same directory.

* Step 3:
    If you read the python code currently has image_name = "5". It can be changed for different images.

* Step 4:
    Run the python code. An output text file named "voltage_for_{image_name}" is produced.

* Step 5:
    Make a voltage source in Ltspice (PWL type) and give the pathname of the text file generated in the previous step.

* Step 6:
    Simulate the voltage source using ".tran 150m" command and observe the waveforms

Thus, from the above steps we are able to convert any 28*28 digit image to it's corresponding spike signals. 

## Reference Paper
This is the [link](https://ieeexplore.ieee.org/document/6894575/figures#figures) of IEEE Paper currently being studied for BTP.
