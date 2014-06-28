import AI
import Test.HUnit
import System.Random

testSigmoid = TestCase( do
    assertEqual "sigmoid( 0 ) should be 0.5 " 0.5 (sigmoid 0) 
    assertBool "sigmoid of negative should be less than 0.5" (0.5 > (sigmoid (-10)))
    assertBool "sigmoid of negative should be more than 0" (0 < (sigmoid (-10)))
    assertBool "sigmoid of positive should be more than 0.5" (0.5 < (sigmoid 10))
    assertBool "sigmoid of positive should be less than 1" (1 > (sigmoid 10))
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
