# LeafClassification
This task involves the use of binary leaf images and extracted features, including shape, margin &amp; texture, to accurately identify 99 species of plants. Leaves, due to their volume, prevalence, and unique characteristics, are an effective means of differentiating plant species.

My current convolutional model performs very well, with nearly 100% accuracy and a .009 logloss on the Kaggle Competition testset. It works by first inputting the images into two convolutional layers, then concatenating the embedding that comes out with the numerical extracted features. Then it goes through one last dense layer before calculating the outputs.

# Training the model

To train the combined model simply run

```
cd src
python kfold_train.py
```

# Visualization

To run the visualization script to see what the neural network is doing in the convolutional layers, run

```
cd src
python visualization.py
```

# JuPyter Kernel

Check out [my JuPyter Kernel](https://www.kaggle.com/abhmul/leaf-classification/keras-convnet-lb-0-0052-w-visualization) outlining this solution on Kaggle!
