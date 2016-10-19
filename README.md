# LeafClassification
This task involves the use of binary leaf images and extracted features, including shape, margin &amp; texture, to accurately identify 99 species of plants. Leaves, due to their volume, prevalence, and unique characteristics, are an effective means of differentiating plant species.

My current convolutional model performs very well, with nearly 100% accuracy and a .009 logloss on the Kaggle Competition testset. It works by first inputting the images into two convolutional layers, then concatenating the embedding that comes out with the numerical extracted features. Then it goes through one last dense layer before calculating the outputs.
