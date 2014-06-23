import System.Random
import System.IO
import Data.List
import Control.Monad
import Control.Concurrent
import Debug.Trace

data Neuron = Neuron {
    weights :: [Double],
    function :: Double -> Double,
    derivative :: Double -> Double
}

instance Show Neuron where
    show = (++) "Neuron " . show . weights

type NeuralLayer = [Neuron]
type NeuralNetwork = [NeuralLayer]

type Seed = StdGen

finiteRandoms :: (Random a) => Int -> Seed -> ([a], Seed)  
finiteRandoms 0 gen = ([], gen)  
finiteRandoms n gen =   
    let (value, newGen) = random gen  
        (restOfList, finalGen) = finiteRandoms (n-1) newGen  
    in (value:restOfList, finalGen) 

sigmoid :: Double -> Double
sigmoid x = (exp x)/(1 + exp x)

sigmoidDerivative :: Double -> Double
sigmoidDerivative x = (exp x)/((exp x + 1)**2)

createSigmoidNeuron :: [Double] -> Neuron
createSigmoidNeuron w = Neuron w sigmoid sigmoidDerivative

createLinearNeuron :: [Double] -> Neuron
createLinearNeuron w = Neuron w id (\x -> 1)

createSigmoidLayer :: [[Double]] -> NeuralLayer
createSigmoidLayer = map createSigmoidNeuron

createLinearLayer :: [[Double]] -> NeuralLayer
createLinearLayer = map createLinearNeuron

initNeuronWeights :: Int -> Seed -> ([Double], Seed)
initNeuronWeights numWeights seed =
    let (r, seed') = finiteRandoms numWeights seed
        weights = map (\x -> x - 0.5 ) r
    in (weights, seed')

initLayerWeights :: Int -> Int -> Seed -> ([[Double]], Seed)
initLayerWeights 0 numWeights seed = ([], seed)
initLayerWeights numNeurons numWeights seed = 
    let (firstWeights, seed0) = initNeuronWeights numWeights seed
        (rest, seed1) = initLayerWeights (numNeurons-1) numWeights seed0
    in ( (firstWeights:rest), seed1 )

activate :: [Double] -> Neuron -> Double
activate i n = 
    let g = function n
        w = weights n
        sigma = sum $ zipWith (*) i w
    in g sigma

backwardActivate :: Neuron -> [Double] -> Double -> Double
backwardActivate n i weightedDelta =
    let g' = derivative n
        w = weights n
        last_in = sum $ zipWith (*) i w
    in (g' last_in) * weightedDelta

createNN3 :: Seed -> Int -> Int -> Int -> NeuralNetwork
createNN3 gen numInput numHidden numOutput = 
    let (hiddenWeights, gen') = initLayerWeights numHidden numInput gen
        (outputWeights, _) = initLayerWeights numOutput numHidden gen'
        hiddenLayer = createSigmoidLayer hiddenWeights
        outputLayer = createSigmoidLayer outputWeights
    in [hiddenLayer, outputLayer]

runNN :: NeuralNetwork -> [Double] -> [Double]
runNN n i = runNNLayers i n

runNNLayers :: [Double] -> [NeuralLayer] -> [Double]
runNNLayers = foldl runNNLayer

-- Calculate list of output per layer
layersOutput :: [Double] -> [NeuralLayer] -> [[Double]]
layersOutput input n = foldl (\outputs l -> ((runNNLayer (head outputs) l):outputs)) [input] n

runNNLayer :: [Double] -> NeuralLayer -> [Double]
runNNLayer = zipWith activate . repeat

updateWeights :: Double -> Neuron -> Double -> [Double] -> Neuron
updateWeights learning_rate n delta inputs = 
    let func = function n
        der = derivative n
        w = weights n
        updatedWeights = zipWith (\weight input -> weight + learning_rate * input * delta) w inputs
    in Neuron updatedWeights func der

updateWeightsLayer :: Double -> NeuralLayer -> [Double] -> [[Double]] -> NeuralLayer
updateWeightsLayer = zipWith3 . updateWeights

updateWeightsNN :: Double -> NeuralNetwork -> [[Double]] -> [[[Double]]] -> NeuralNetwork
updateWeightsNN = zipWith3 . updateWeightsLayer

sumProduct :: [Double] -> [Double] -> Double
sumProduct = (sum .) . zipWith (*)

weightSigmaLayer :: [Double] -> NeuralLayer -> [Double]
weightSigmaLayer delta layer = let
    -- clean this up, transpose?
    numWeights = length (weights (layer !! 0))
    w = map (\(l,i) -> map (\n -> (weights n) !! i) l) $ zip (replicate numWeights layer) [0..]
    in zipWith sumProduct w (replicate numWeights delta)

layerDelta :: NeuralLayer -> [[Double]] -> [Double] -> [Double]
layerDelta = zipWith3 backwardActivate

outputsToInputs :: [[Double]] -> [[[Double]]]
outputsToInputs = init . (map repeat)

backProp :: [NeuralLayer] -> [Double] -> [Double] -> Double -> NeuralNetwork
backProp n input target learning_rate = 
        -- O = outputMatrix
    let outputs = layersOutput input n
        -- I = inputMatrix
        inputs = outputsToInputs $ reverse outputs
        -- Result = output row = head O
        result = head outputs
        outputLayer = last n
        outputError = zipWith (-) target result
        -- For each output node i
        --    Di = g'( Ii ) * ( Yi - Ai )
        delta = zipWith3 backwardActivate outputLayer (last inputs) outputError
        hiddenLayers = init n
        -- For l = L - 1 to 1
        (deltas,_,_) = foldr (\l acc -> 
            let (deltas,inputs,prevLayer) = acc
                (input:inputs') = inputs
                prevDelta = head deltas
                -- For each node j in layer l
                --     Dj = g'(Ij) * Σi (Di * Wji)
                delta = zipWith3 backwardActivate l input (weightSigmaLayer prevDelta prevLayer)
                deltas' = delta:deltas
            in (deltas',inputs',l) ) ([delta],(reverse inputs),outputLayer) hiddenLayers
        -- Wji = Wji + alpha * Ii * Di
    in updateWeightsNN learning_rate n deltas inputs 

average :: (Real a, Fractional b) => [a] -> b
average xs = realToFrac (sum xs) / genericLength xs

norm :: [Double] -> Double
norm = sqrt . sum . map (** 2)

percentError :: [Double] -> [Double] -> Double
percentError target output = let
    e = zipWith (-) target output
    in 100 * ( (norm e) / (norm target) )

squareError :: [Double] -> [Double] -> Double
squareError target output = (** 2) $ norm $ zipWith (-) target output

rmse :: [[Double]] -> [[Double]] -> Double
rmse targets outputs = sqrt $ average $ zipWith squareError targets outputs

trainCos = do
    gen <- getStdGen
    let n = createNN3 gen 1 10 1
    trainedN <- trainCosLoop n 10000 0 1
    return trainedN

trainCosLoop :: NeuralNetwork -> Int -> Int -> Double -> IO NeuralNetwork
trainCosLoop n times timesTrained lastRmse = do
    if ( times == timesTrained )
        then do return n
        else do
            gen <- newStdGen
            r <- randomIO :: IO Double
            let input = r * pi / 2
            let target = [cos input]
            let n' = backProp n [input] target 1.5
            let e = validateCosLoop gen n 1000 [] []
            putStrLn $ "Root mean square error: " ++ (show e) ++ " (" ++ (show timesTrained) ++ ")"
            trainCosLoop n' times (timesTrained + 1) e

validateCosLoop :: Seed -> NeuralNetwork -> Int -> [[Double]] -> [[Double]] -> Double
validateCosLoop gen n 0 outputs targets = 
    rmse targets outputs
    
validateCosLoop gen n times outputs targets =
    let (r,gen') = random gen
        input = r * pi / 2
        target = [cos input]
        output = runNN n [input]
        targets' = (target:targets)
        outputs' = (output:outputs)
    in validateCosLoop gen' n (times-1) outputs' targets'

guessGame = do 
    hSetBuffering stdin LineBuffering
    gen <- getStdGen
    let n = createNN3 gen 2 10 1
    guessGameLoop n

guessGameLoop :: NeuralNetwork -> IO ()
guessGameLoop n = do
    putStr "Give me two numbers: "
    inputString <- getLine
    when ( not $ null inputString ) $ do
        let numberStrings = words inputString
        let input = map ((/ 100) . read) numberStrings
        let output = runNN n input
        putStrLn $ "My guess: " ++ (show (round (100 * head output)))
        putStr "What's the correct result? "
        resultString <- getLine
        when ( not $ null resultString ) $ do 
            let result = (/ 100) $ read resultString
            let n' = backProp n input [result] 5
            guessGameLoop n'
