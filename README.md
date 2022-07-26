# nia-sound-model

## One-cycle test

This is the code for one-cycle test in NIA data collection projects.

A simple neural network model is used to predict AHI from sound data. The input to the model is Mel Spectrogram and the model outputs classification result which predict if AHI value is higher than 15 or not.

The data should be split to test and train datasets. The path structure for the data is as follows.

```
data
  ├─ test
  │   ├─ sound
  │   │   └─ npy files
  │   └─ label
  │       └─ json files
  └─ train
      ├─ sound
      │   └─ npy files
      └─ label
          └─ json files

```
