import AI
import Text.CSV
import Data.List
import System.Random
import Text.Printf

dataRowToTestData :: Record -> ([Double],[Double])
dataRowToTestData row =
    case elemIndex (row !! 0) ["M","F","I"] of
        Nothing -> ([],[])
        Just i -> let
            sex = (fromIntegral i) / 3
            len = read (row !! 1)
            diameter = read (row !! 2)
            height = read (row !! 3 )
            wholeWeight = read (row !! 4)
            shuckedWeight = read (row !! 5)
            visceraWeight = read (row !! 6)
            shellWeight = read (row !! 7)
            rings = read (row !! 8)
            in ([sex,len,diameter,height,wholeWeight,shuckedWeight,visceraWeight,shellWeight],[rings/30])

percentCorrect :: [ [ Double ] ] -> [ [ Double ] ] -> Double
percentCorrect targets predictions = let
    numCorrect = length $ filter ( \(target,prediction) -> (round (30 * (head target))) == (round (30 * (head prediction))) ) $ zip targets predictions
    in (fromIntegral (100 * numCorrect)) / (genericLength targets)

validateNNDataset :: NeuralNetwork -> [([Double],[Double])] -> Double
validateNNDataset n ds = let
    (targets,predictions) = runNNDataset n ds
    in percentCorrect targets predictions

main = do
    result <- parseCSVFromFile "abalone.data"
    case result of
        Left e -> putStrLn $ show e
        Right csv -> do
            -- last record is empty due to final \n, so we use (init csv)
            let dataSet = map dataRowToTestData (init csv)
            -- use 3133 rows for training and the rest for testing the performance
            let (trainDS,testDS) = splitAt 3133 dataSet
            let (n,gen') = createNN3 (mkStdGen 1) 8 22 1
            let n' = trainNNDataset 0.14 n trainDS
            -- Output the percentage of correctly classified abalones
            putStrLn $ printf "%.2f%%" $ validateNNDataset n' testDS
