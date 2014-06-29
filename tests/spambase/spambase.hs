import AI
import Text.CSV
import Data.List
import System.Random
import Text.Printf

dataRowToTestData :: Record -> ([Double],[Double])
dataRowToTestData row = let
    nums = map (\(x,i) ->
        case i of
            -- convert 55th,56th and 57th column into range 0..1
            -- otherwise large values cause troubles
            55 -> ((read x) / 1200)
            56 -> ((read x) / 10000)
            57 -> ((read x) / 20000)
            _ -> read x
        ) (zip row [0..])
    input = init nums
    output = [ last nums ]
    in (input,output)

percentCorrect :: [ [ Double ] ] -> [ [ Double ] ] -> Double
percentCorrect targets predictions = let
    numCorrect = length $ filter ( \(target,prediction) -> (round (head target)) == (round (head prediction)) ) $ zip targets predictions
    in (fromIntegral (100 * numCorrect)) / (genericLength targets)

validateNNDataset :: NeuralNetwork -> [([Double],[Double])] -> Double
validateNNDataset n ds = let
    (targets,predictions) = runNNDataset n ds
    in percentCorrect targets predictions

main = do
    result <- parseCSVFromFile "spambase.data"
    case result of
        Left e -> putStrLn $ show e
        Right csv -> do
            -- last record is empty due to final \n, so we use (init csv)
            let dataSet = map dataRowToTestData (init csv)
            -- use 75% of rows for training and the rest for testing the performance
            let (trainDS,testDS) = splitAt 3451 dataSet
            let (n,gen') = createNN3 (mkStdGen 1) 57 60 1
            let n' = trainNNDataset 0.1 n trainDS
            putStrLn $ printf "%.2f%%" $ validateNNDataset n' testDS
