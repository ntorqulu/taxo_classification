# Project Results

This folder cotains the results of the different trainings that we have performed for each model.

In summary the project has evaluated three different factor

- Model used. (basic, enhanced_mlp, cnn, nanni_cnn1...)

- Dataset used. Refering to the filtering process in which three different methodologies can be used. First to train with all the labels present in the dataset. Second, to merge the less abundand categories into merged labels. And third to remove those sequences of labels that do not have more than n records in the database.

- Level which the model has been trained to predict. Kingdom, phylum...


In this folder the results are structured in the following way:

- Results
    |- logs
        | - copied folder for TensorFlow 
    |- result of each case
        |- README.md
        |- json
        |- best checkout