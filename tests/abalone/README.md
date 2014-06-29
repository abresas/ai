Abalone
=======

Predicting the age of abalone from physical measurements.

The dataset has 8 attributes of abalones and a final column which is the number of their rings (9 columns total). The age of an abalone is 1.5 * rings.

abalone.hs is a haskell script that tests the performance of the library on the dataset. It trains a neural network and then prints the percentage it can successfully predict the exact number of rings.

To run the test, first install the AI library:

    $ cd (ai-project-root) 
    $ cabal build
    $ cabal install

And then run the abalone.hs script:

    $ cd tests/abalone
    $ runhaskell abalone.hs
    25.67%

Test Set Performance
====================

The neural network has 25.67% success rate in predicting the exact number of rings, with 22 hidden nodes and 0.14 learning rate.

The number of hidden nodes and learning rate was found with simple trial and error.

The original authors had 26.25% with a Cascade-Correlation Neural Network with 5 hidden nodes.

Dataset
=======

Dataset taken from UCI Machine Learning Repository:

http://archive.ics.uci.edu/ml

Page for abalone dataset:

http://archive.ics.uci.edu/ml/datasets/Abalone
