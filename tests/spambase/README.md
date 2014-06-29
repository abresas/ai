Spambase
========

Classifying Email as Spam or Non-Spam

The dataset has 57 attributes of email texts and a final one classifying it as a spam or non-spam.

spambase.hs is a haskell script that tests the performance of the AI library on the dataset. It trains a neural network and then prints the percentage it can classify correctly.

To run the test, first install the AI library:

    $ cd (ai-project-root) 
    $ cabal build
    $ cabal install

And then run the spambase.hs script:

    $ cd tests/spambase
    $ runhaskell spambase.hs
    100%

Test Set Performance
====================

The neural network has 100% success rate in classifying the validation data.

The original authors reported ~7% misclassification error.

Dataset
=======

Dataset taken from UCI Machine Learning Repository:

http://archive.ics.uci.edu/ml

Page for spambase dataset:

http://archive.ics.uci.edu/ml/datasets/Spambase

The dataset records were shuffled with the shuf unix command before being added to this repository.
