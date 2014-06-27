Intro
=====

A small library for neural networks written in haskell.

I made it just for fun and it's completely experimental.

It probably needs a better name :)

Installation
============

Download or clone the project and run:

    $ cabal configure
    $ cabal install

You will need to have cabal-install package installed in your system.

Usage
=====

Create a 3-layer neural network n, having 1 input, 10 hidden, and 1 output nodes, with initial seed gen:

    >>> gen <- newStdGen
    >>> let (n,gen') = createNN3 gen 1 10 1

*createNN3* like most functions in these examples, because they need randomness, require a seed parameter and return a pair with the result and the next state of the seed.

Run the neural network with *runNN*

    >>> runNN n [0]
    [0.4850840962158099]

Train the neural network to learn the cosine in [0..1]:

    >>> let (n',gen'') = trainNN n (\xs -> [ cos $ head xs ] ) 0.9 1000 gen'

The training function must have type [ Double ] -> [ Double ],
so in the 2nd parameter we transform cos into accepting and returning a list.

0.9 is the learning rate, and 1000 is the times to repeat the training. 

The training function will be called with random numbers between 0 and 1.

Now we can run the neural network again for varius inputs to see the output:

    >>> runNN n' [0]
    [0.8840487237322846]
    >>> runNN n' [0.5]
    [0.8376510812538144]
    >>> runNN n' [1]
    [0.784193327107528]

We can train the neural network more times to gain better results:

    >>> let (n',gen'') = trainNN n (\xs -> [ cos $ head xs ] ) 0.9 5000 gen'
    >>> runNN n' [1]
    [0.5629223322703413]

which is much closer to the actual value of cos 1 = 0.54.

*runNN1* takes a neural network that has 1 input and 1 output, and a double to be used as input.

    >>> runNN1 n' 0
    0.9846716920261342

*runNN1* can also be seen as a function that transforms neural network into a function accepting a double:

    >>> let mycos = runNN1 n'
    >>> :t mycos
    mycos :: Double -> Double
    >>> mycos 0
    0.9846716920261342

We can validate how well our neural network has learned the given function by calling *validateNN*:

    >>> let (e,_) = validateNN n' (\xs -> [ cos $ head xs ] ) 1000 gen''
    >>> e
    1.2427629031998429e-2
    
*validateNN* returns the Root Mean Square Error of the results of the neural network compared to the results of the target function. Each result and target output is normalized before calculating, so the error is always returned as a Double (even if the neural network has multiple inputs and outputs).
