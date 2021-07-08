# CNN_classifier
by Bernhard Irsigler

This is code for a classification task of spin Hamiltonians into classes of different quantum annealing fidelity, i.e., of classes of different hardness. As this is a quantum problem and the time evolution is unitary, there is only a finite number of different fidelities. Those are categorized into classes even though the fidelities are real numbers. The outcome is much more promising than regression however also more demanding in a computational sense.

The data itself is not part of this repository in order to safe space. 

CNN_classifier.ipynb contains a first model to solve the task
CNN_classifier_HPtuning.ipynb does hyperparameter tuning
CNN_classifier_module contains the best model found through hyperparameter tuning as an importable module
CNN_classifier_sampleData.ipynb tries to solve the task with incomplete data
CNN_classifier_dropData.ipynb tries to solve the task by recreating missing data through the kNN algorithm
