{-|
    Module:         AI
    Description:    Create and train neural networks
    Copyright:      (c) Aleksis Brezas, 2014 
    License:        MIT

    Stability:      experimental
    Portability:    POSIX
-}
module AI ( 
    -- * Types
    Neuron, NeuralLayer, NeuralNetwork, 
    -- * Neural network
    createNN3, runNN, runNN1, runNN2,
    getInputSize, getOutputSize,
    -- ** Layer
    createSigmoidLayer, createLinearLayer, initLayerWeights,
    -- ** Neuron
    createSigmoidNeuron, createLinearNeuron, initNeuronWeights, activate,
    -- * Training
    backProp, trainNN, validateNN, testTrainNN
    ) where

import System.Random
import System.IO
import Data.List
import Control.Monad
import Control.Concurrent
import Debug.Trace

-------------------------- GENERIC -----------------------------

-- |Given a function f and an initial value x,
-- call f repetitively n times, producing the result f(f(..f(x)..)).
loop :: Int -> (a -> a) -> a -> a
loop n = ((!! n) .) . iterate

--------------------------- MATH --------------------------------

-- |Calculate average (or mean) of list of reals.
-- = ( 1 / n ) * sum
mean :: Floating a => [a] -> a
mean xs = (sum xs) / genericLength xs

-- |Normalize vector
norm :: Floating a => [a] -> a
norm = sqrt . sum . map (** 2)

-- |Sum the inner product of two lists
sumProduct :: Num a => [a] -> [a] -> a
sumProduct = (sum .) . zipWith (*)

------------------------ STATISTICS -----------------------------

-- |Calculate error of prediction vector as percentage.
percentError :: Floating a => [a] -> [a] -> a
percentError target output = let
    e = zipWith (-) target output
    in 100 * ( (norm e) / (norm target) )

-- |Calculate mean square error of a list of targets and list of predictions.
--
-- The result has a magnitude of the inputs squared, so for a result of same magnitude as inputs, call rmse.
mse :: Floating a => [a] -> [a] -> a
mse = (mean .) . zipWith ( ( (** 2) . ) . (-) )

-- |Calculate root mean square error of a list of targets and list of predictions.
--
-- Targets and predictions must be of type double.
rmse :: Floating a => [a] -> [a] -> a
rmse = (sqrt .) . mse

-- |Calculate root mean square error of normalized targets and normalized predictions.
--
-- This is like RMSE but first normalizes targets and predictions
normRMSE :: Floating a => [[a]] -> [[a]] -> a
normRMSE t p = rmse (map norm t) (map norm p)

--------------------------- RANDOM -----------------------------

type Seed = StdGen

-- |Generate a list of finite randoms.
finiteRandoms :: (Random a) => Int -> Seed -> ([a], Seed)  
finiteRandoms 0 gen = ([], gen)  
finiteRandoms n gen =   
    let (value, newGen) = random gen  
        (restOfList, finalGen) = finiteRandoms (n-1) newGen  
    in (value:restOfList, finalGen) 

------------------------- NEURAL NETWORK ------------------------------

data Neuron = Neuron {
    weights :: [Double],
    function :: Double -> Double,
    derivative :: Double -> Double
}

instance Show Neuron where
    show = (++) "Neuron " . show . weights

type NeuralLayer = [Neuron]
type NeuralNetwork = [NeuralLayer]

-- |Feed-forward activate neuron with given input.
activate :: [Double] -> Neuron -> Double
activate i n = 
    let g = function n
        w = weights n
        sigma = sum $ zipWith (*) i w
    in g sigma

-- |Feed-backward activate neuron with given input and weighted delta.
backwardActivate :: Neuron -> [Double] -> Double -> Double
backwardActivate n i weightedDelta =
    let g' = derivative n
        w = weights n
        last_in = sum $ zipWith (*) i w
    in (g' last_in) * weightedDelta

-- |Logistic function :: R -> [0..1]
sigmoid :: Double -> Double
sigmoid x = (exp x)/(1 + exp x)

-- |Derivative of logistic function.
sigmoidDerivative :: Double -> Double
sigmoidDerivative x = (exp x)/((exp x + 1)**2)

-- |Create a neuron with sigmoid activation using given weights.
createSigmoidNeuron :: [Double] -> Neuron
createSigmoidNeuron w = Neuron w sigmoid sigmoidDerivative

-- |Create a linear neuron with given weights.
--
-- May be useful for output layer.
--
-- Curently uses identity function as activation.
createLinearNeuron :: [Double] -> Neuron
createLinearNeuron w = Neuron w id (\x -> 1)

-- |Create a layer of sigmoid neurons.
createSigmoidLayer :: [[Double]] -> NeuralLayer
createSigmoidLayer = map createSigmoidNeuron

-- |Create a layer of linear neurons.
--
-- May be useful as output layer.
createLinearLayer :: [[Double]] -> NeuralLayer
createLinearLayer = map createLinearNeuron

-- |Return a specific number of initial weights.
--
-- The weights are adjusted to be a random number between -0.5 and 0.5.
--
-- These are considered \"acceptable\" weights, but
-- many studies have proved that there are much better initial values.
--
-- This needs some work.
initNeuronWeights :: Int -> Seed -> ([Double], Seed)
initNeuronWeights numWeights seed =
    let (r, seed') = finiteRandoms numWeights seed
        weights = map (\x -> x - 0.5 ) r
    in (weights, seed')

-- |Use initNeuronWeights to generate initial weights for whole layer.
initLayerWeights :: Int -> Int -> Seed -> ([[Double]], Seed)
initLayerWeights 0 numWeights seed = ([], seed)
initLayerWeights numNeurons numWeights seed = 
    let (firstWeights, seed0) = initNeuronWeights numWeights seed
        (rest, seed1) = initLayerWeights (numNeurons-1) numWeights seed0
    in ( (firstWeights:rest), seed1 )

-- |Create a 3-layer neural network (input, hidden, output).
--
-- The neural network doesnt store representation of the input layer,
-- since it doesnt have any weights.
createNN3 :: Seed -> Int -> Int -> Int -> (NeuralNetwork,Seed)
createNN3 gen numInput numHidden numOutput = 
    let (hiddenWeights, gen') = initLayerWeights numHidden numInput gen
        (outputWeights, gen'') = initLayerWeights numOutput numHidden gen'
        hiddenLayer = createSigmoidLayer hiddenWeights
        outputLayer = createSigmoidLayer outputWeights
    in ([hiddenLayer, outputLayer],gen'')

-- | Return size of input layer.
getInputSize :: NeuralNetwork -> Int
getInputSize = length . weights . head . head

-- | Return size of output layer.
getOutputSize :: NeuralNetwork -> Int
getOutputSize = length . last

-- |Run a multi-layer neural network and return its output.
runNN :: NeuralNetwork -> [Double] -> [Double]
runNN n i = runNNLayers i n

-- |Run a neural network with 1 input and 1 output
runNN1 :: NeuralNetwork -> Double -> Double
runNN1 n i = head $ runNNLayers [i] n

-- |Run a neural network with 2 inputs and 1 output
runNN2 :: NeuralNetwork -> Double -> Double -> Double
runNN2 n i1 i2 = head $ runNNLayers [i1,i2] n

-- |Run each layer of this neural network and return the last as output.
runNNLayers :: [Double] -> [NeuralLayer] -> [Double]
runNNLayers = foldl runNNLayer

-- |Run layer and return output.
runNNLayer :: [Double] -> NeuralLayer -> [Double]
runNNLayer = zipWith activate . repeat

-- |Like runNN but returns list of output per layer (output matrix).
layersOutput :: [Double] -> [NeuralLayer] -> [[Double]]
layersOutput input n = foldl (\outputs l -> outputs ++ [(runNNLayer (last outputs) l)]) [input] n

------------------------ NN LEARNING --------------------------------

-- |Update weights of a neuron given learning rate, delta and input.
--
-- The formula is Wij = Wij + l * aj * Di.
updateWeights :: Double -> Neuron -> Double -> [Double] -> Neuron
updateWeights learning_rate n delta inputs = 
    let func = function n
        der = derivative n
        w = weights n
        updatedWeights = zipWith (\weight input -> weight + learning_rate * input * delta) w inputs
    in Neuron updatedWeights func der

-- |Update weights of layer given learning rate, list of neurons, list of deltas, list of input to neuron.
updateWeightsLayer :: Double -> NeuralLayer -> [Double] -> [[Double]] -> NeuralLayer
updateWeightsLayer = zipWith3 . updateWeights

-- |Update weights on neural network after back propagation.
--
-- Parameters: learning rate, neural network, matrix of deltas (per neuron), matrix of input to neuron
updateWeightsNN :: Double -> NeuralNetwork -> [[Double]] -> [[[Double]]] -> NeuralNetwork
updateWeightsNN = zipWith3 . updateWeightsLayer

-- |Calculates weighted deltas for a layer.
--
-- Equal to:
--
--      for each neuron j: Σi Wij Dj
--
-- Where:
--
--      * Wij weights from i to j
--      * Dj delta of neuron j
--      * Σi sum for each connected neuron i
weightSigmaLayer :: [Double] -> NeuralLayer -> [Double]
weightSigmaLayer delta layer = let
    -- using tranpose we get weights grouped by source node
    -- instead of weights grouped by destination node (as usual)
    w = transpose $ map (\n -> (weights n)) layer
    in zipWith sumProduct w (repeat delta)

-- |Calculate delta for each neuron in layer.
layerDelta :: NeuralLayer -> [[Double]] -> [Double] -> [Double]
layerDelta = zipWith3 backwardActivate

-- |Transform matrix of outputs to matrix of inputs.
outputsToInputs :: [[Double]] -> [[[Double]]]
outputsToInputs = init . (map repeat)

-- |Calculate delta for each neuron in each layer of neural network.
layersDelta :: [[[Double]]] -> [Double] -> [[Double]] -> [NeuralLayer] -> [[Double]]
layersDelta inputMatrix prevError acc [] = acc
layersDelta inputMatrix prevError acc layers = let
    (l:layers') = layers
    (input:inputMatrix') = inputMatrix
    delta = zipWith3 backwardActivate l input prevError
    layerError = weightSigmaLayer delta l
    in layersDelta inputMatrix' layerError (delta:acc) layers'

-- |Backpropagate multi-layer network using a single training sample (input,target) 
--
-- learning_rate is the speed of learning, typically in the range [0..1], usually 0.1.
--
-- Large learning_rate allows for faster learning, but smaller values will typically result in better learning.
--
-- A training function will call this function multiple times
-- to get a network with improved weights every time.
--
-- Example teaching sine function:
--
-- > let (n,_) = createNN3 seed 1 10 1
-- > let n' = backProp n [0] [0] 0.1
-- > let n'' = backProp n' [pi/2] [1] 0.1
--
-- Different learning rates may be used during the training of the same network,
-- for example by passing large values at the beginning and smaller ones later.
--
-- TODO: momentum
backProp :: [NeuralLayer] -> [Double] -> [Double] -> Double -> NeuralNetwork
backProp n input target learning_rate = let
    -- Calculate output per layer
    outputs = layersOutput input n
    -- Transform output per layer into input per layer
    inputs = outputsToInputs outputs
    -- Last output is the result
    result = last outputs
    -- Calculate error of output layer
    outputError = zipWith (-) target result
    -- Calculate delta for each layer
    deltas = layersDelta (reverse inputs) outputError [] (reverse n)
    -- Use deltas to update weights
    in updateWeightsNN learning_rate n deltas inputs 

-- |Train a neural network to learn a function ([Double] -> [Double])
-- The function will be called with random input in [0..1]
-- and is expected to have output in [0..1]
trainNN :: NeuralNetwork -> ([Double] -> [Double]) -> Double -> Int -> Seed -> (NeuralNetwork,Seed)
trainNN nn _ _ 0 gen = (nn, gen)
trainNN nn func learningRate times gen = let
    numInput = getInputSize nn
    (input,gen') = finiteRandoms numInput gen :: ([Double],Seed)
    target = func input
    nn' = backProp nn input target learningRate
    in trainNN nn' func learningRate (times-1) gen'

-- |Validate how well the neural network has learned the given function.
validateNN :: NeuralNetwork -> ([Double] -> [Double]) -> Int -> Seed -> (Double,Seed)
validateNN nn func times gen = let
    (targets,outputs,gen') = loop times ( \(t,o,g) -> let
        numInput = getInputSize nn
        (input,g') = finiteRandoms numInput g :: ([Double],Seed)
        target = func input
        output = runNN nn input
        in (target:t,output:o,g') ) ([],[],gen)
    in (normRMSE targets outputs, gen')

-- |Train neural network and return the validation results directly.
--
-- Useful for testing the library.
testTrainNN :: NeuralNetwork -> ([Double] -> [Double]) -> Double -> Int -> Seed -> (Double,Seed)
testTrainNN nn func learningRate times gen = let
    (nn',gen') = trainNN nn func learningRate times gen
    in validateNN nn' func 1000 gen'
