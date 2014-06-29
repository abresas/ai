import AI
import Control.Exception
import Control.Monad
import Test.HUnit
import System.Random

instance Eq ErrorCall where
    x == y = (show x) == (show y)

assertException :: (Exception e, Eq e) => String -> e -> IO a -> IO ()
assertException msg ex action =
    handleJust isWanted (const $ return ()) $ do
        action
        assertFailure msg
  where isWanted = guard . (== ex) 

assertError msg ex f = 
    assertException msg (ErrorCall ex) $ evaluate f

testSigmoid = TestCase( do
    assertEqual "sigmoid( 0 ) should be 0.5 " 0.5 (sigmoid 0) 
    assertBool "sigmoid of negative should be less than 0.5" (0.5 > (sigmoid (-10)))
    assertBool "sigmoid of negative should be more than 0" (0 < (sigmoid (-10)))
    assertBool "sigmoid of positive should be more than 0.5" (0.5 < (sigmoid 10))
    assertBool "sigmoid of positive should be less than 1" (1 > (sigmoid 10))
    assertBool "sigmoid of positive up to 699 should be larger or equal to 1" (1 >= (sigmoid 699))
    assertError "large input to sigmoid should cause error" "Too large input to neuron. Try scaling down the neural network input." (sigmoid 700)
    )

testCreateNN3 = TestCase( do
    gen <- newStdGen
    let (n,gen') = createNN3 gen 2 10 5
    assertEqual "createNN3 should create a 3-layer NN" 3 $ getNumLayers n
    assertEqual "createNN3 should create a NN with specified input size" 2 $ getInputSize n
    assertEqual "createNN3 should create a NN with specified output size" 5 $ getOutputSize n
    let ws = concat $ map (\l -> concat $ map (\n -> weights n) l) n 
    let wCorrect = all (\w -> and [ (w > (- 0.5)), (w < 0.5) ] ) ws
    assertBool "initially all weights should have values between -0.5 and 0.5" wCorrect
    )

tests = TestList[ testSigmoid, testCreateNN3 ]
