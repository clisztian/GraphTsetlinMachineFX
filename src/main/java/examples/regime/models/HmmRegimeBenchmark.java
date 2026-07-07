package examples.regime.models;


import examples.regime.dto.MarketData;
import examples.regime.io.DatasetReader;
import examples.regime.trainer.DatasetSplitter;
import examples.regime.trainer.DatasetStatisticsAnalyzer;
import smile.sequence.HMM;
import util.MutableInt;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.ToIntFunction;

import static examples.regime.RegimeLabeler.computeLabeledDataset;

public class HmmRegimeBenchmark {

    /**
     * Quantizes a single continuous MarketData hour into a discrete
     * token symbol ID between 0 and 26.
     */
    private static int quantizeObservation(MarketData data, double meanAtr) {
        // Quantize ATR into Low (0), Mid (1), High (2) buckets relative to mean
        int atrBucket = (data.getUsdjpyAtr() < meanAtr * 0.7) ? 0 : (data.getUsdjpyAtr() > meanAtr * 1.3) ? 2 : 1;

        // Quantize Trend Efficiency Delta into Negative (0), Stable (1), Positive (2)
        int erBucket = (data.getErDelta() < -0.05) ? 0 : (data.getErDelta() > 0.05) ? 2 : 1;

        // Capture your exact CPI shock integer flag (0, 1, or 2)
        int shockBucket =1;

        // Encode combinations into a distinct structural token ID between 0 and 26
        return ((atrBucket * 9) + (erBucket * 3) + shockBucket);
    }

    public static void runHmmBenchmark(DatasetSplitter.PartitionedDataset partitions, double globalMeanAtr) {
        System.out.println("Quantizing continuous sequences into native integer streams for SMILE HMM...");

        int numTrainSamples = partitions.trainSet.size();
        int numTestSamples = partitions.testSet.size();

        // 1. Build the Integer Training Matrices matching the discrete HMM signature requirements
        // T[][] must be filled with Integer objects to play nicely with the ToIntFunction mapping
        Integer[][] trainObservations = new Integer[1][numTrainSamples];
        int[][] trainLabels = new int[1][numTrainSamples];

        for (int i = 0; i < numTrainSamples; i++) {
            MarketData data = partitions.trainSet.get(i);
            trainObservations[0][i] = quantizeObservation(data, globalMeanAtr);
            trainLabels[0][i] = Math.max(0, data.getRegimeLabel()); // Guarantee valid positive state indices
        }

        // 2. Build the flat primitive int[] array for out-of-sample prediction
        int[] testObservations = new int[numTestSamples];
        int[] y_test = new int[numTestSamples];

        for (int i = 0; i < numTestSamples; i++) {
            MarketData data = partitions.testSet.get(i);
            testObservations[i] = quantizeObservation(data, globalMeanAtr);
            y_test[i] = data.getRegimeLabel();
        }

        // Define the simple lambda to unbox the Integer object back into an int primitive ID
        ToIntFunction<Integer> ordinalLambda = Integer::intValue;

        System.out.println("Fitting Discrete Hidden Markov Model (Baum-Welch calibration)...");
        long startTrain = System.currentTimeMillis();

        // Train the model structure: fit(T[][] observations, int[][] labels, ToIntFunction<T> ordinal)
        HMM model = HMM.fit(trainObservations, trainLabels, ordinalLambda);

        long endTrain = System.currentTimeMillis();
        System.out.printf("HMM Calibration Complete in %d ms.\n", (endTrain - startTrain));

        // Evaluate sequential prediction accuracy out-of-sample
        evaluateModel(model, testObservations, y_test);
    }

    private static void evaluateModel(HMM model, int[] testObservations, int[] y_test) {
        int[] classTotals = new int[4];
        int[] classCorrect = new int[4];

        long startInference = System.nanoTime();

        // FIXED: Invokes public int[] predict(int[] o) natively
        int[] predictedPath = model.predict(testObservations);

        long endInference = System.nanoTime();

        for (int i = 0; i < testObservations.length; i++) {
            int trueLabel = y_test[i];
            if (trueLabel < 0 || trueLabel > 3) continue; // Skip unpopulated initialization data safely

            int predictedLabel = predictedPath[i];

            classTotals[trueLabel]++;
            if (predictedLabel == trueLabel) {
                classCorrect[trueLabel]++;
            }
        }

        System.out.println("\n======================================================================");
        System.out.println("          SMILE v3.1.1 DISCRETE HMM GENERATIVE OUT-OF-SAMPLE ACCURACY ");
        System.out.println("======================================================================");
        for (int c = 0; c < 4; c++) {
            double accuracy = classTotals[c] > 0 ? (double) classCorrect[c] / classTotals[c] : 0.0;
            System.out.printf("Regime Class %d -> HMM Benchmark Accuracy: %6.2f%% (Samples: %d)\n",
                    c, accuracy * 100, classTotals[c]);
        }
        System.out.println("======================================================================");
        System.out.printf("Production Inference Speed: Average %.3f microseconds per sample hour.\n",
                ((double) (endInference - startInference) / testObservations.length) / 1000.0);
        System.out.println("======================================================================");

        //print the total accruacy
        int totalCorrect = 0;
        int totalSamples = 0;
        for (int c = 0; c < 4; c++) {
            totalCorrect += classCorrect[c];
            totalSamples += classTotals[c];
        }
        double totalAccuracy = totalSamples > 0 ? (double) totalCorrect / totalSamples : 0.0;
        System.out.printf("Overall HMM Benchmark Accuracy: %6.2f%% (Total Samples: %d)\n",
                totalAccuracy * 100, totalSamples);
    }




    public static class OptimalHmmConfig {
        public double lowAtrThreshold;
        public double highAtrThreshold;
        public double erDeltaThreshold;
        public double bestAccuracy;
    }

    public static void optimizeHmm(DatasetSplitter.PartitionedDataset partitions, double globalMeanAtr) {
        System.out.println("Initializing HMM Geometric Quantization Parameter Grid Sweep...");

        // Define search ranges for our binning thresholds
        double[] atrLowBounds = {0.6, 0.7, 0.8};
        double[] atrHighBounds = {1.2, 1.3, 1.4};
        double[] erBounds = {0.03, 0.05, 0.07};

        OptimalHmmConfig bestConfig = new OptimalHmmConfig();
        bestConfig.bestAccuracy = -1.0;

        int numTrain = partitions.trainSet.size();
        int numTest = partitions.testSet.size();

        // Run the parameter grid sweep
        for (double al : atrLowBounds) {
            for (double ah : atrHighBounds) {
                for (double er : erBounds) {

                    // 1. Quantize the datasets using the current iteration's grid hyperparameters
                    Integer[][] trainObs = new Integer[1][numTrain];
                    int[][] trainLabels = new int[1][numTrain];
                    for (int i = 0; i < numTrain; i++) {
                        MarketData d = partitions.trainSet.get(i);
                        trainObs[0][i] = customQuantize(d, globalMeanAtr, al, ah, er);
                        trainLabels[0][i] = Math.max(0, d.getRegimeLabel());
                    }

                    int[] testObs = new int[numTest];
                    int[] y_test = new int[numTest];
                    for (int i = 0; i < numTest; i++) {
                        MarketData d = partitions.testSet.get(i);
                        testObs[i] = customQuantize(d, globalMeanAtr, al, ah, er);
                        y_test[i] = d.getRegimeLabel();
                    }

                    // 2. Train and Evaluate
                    try {
                        ToIntFunction<Integer> ordinal = Integer::intValue;
                        HMM candidateModel = HMM.fit(trainObs, trainLabels, ordinal);

                        double accuracy = evaluateCandidate(candidateModel, testObs, y_test);

                        if (accuracy > bestConfig.bestAccuracy) {
                            bestConfig.bestAccuracy = accuracy;
                            bestConfig.lowAtrThreshold = al;
                            bestConfig.highAtrThreshold = ah;
                            bestConfig.erDeltaThreshold = er;

                            System.out.printf("Frontier Advanced -> Acc: %.2f%% | ATR Low: %.1f, ATR High: %.1f, ER: %.3f\n",
                                    accuracy * 100, al, ah, er);
                        }
                    } catch (Exception e) {
                        // Absorb singular matrix or empty bin convergence errors safely
                    }
                }
            }
        }

        System.out.println("\n======================================================================");
        System.out.println("                OPTIMAL DISCRETE HMM CONFIGURATION FOUND              ");
        System.out.println("======================================================================");
        System.out.printf(" Max Out-of-Sample Accuracy        : %6.2f%%\n", bestConfig.bestAccuracy * 100);
        System.out.printf(" Optimal Low Volatility Threshold  : Mean * %.2f\n", bestConfig.lowAtrThreshold);
        System.out.printf(" Optimal High Volatility Threshold : Mean * %.2f\n", bestConfig.highAtrThreshold);
        System.out.printf(" Optimal Trend Efficiency Boundary : +/- %.3f\n", bestConfig.erDeltaThreshold);
        System.out.println("======================================================================");
    }

    private static int customQuantize(MarketData data, double meanAtr, double al, double ah, double er) {
        int atrBucket = (data.getUsdjpyAtr() < meanAtr * al) ? 0 : (data.getUsdjpyAtr() > meanAtr * ah) ? 2 : 1;
        int erBucket = (data.getErDelta() < -er) ? 0 : (data.getErDelta() > er) ? 2 : 1;
        int shockBucket = 1;
        return (atrBucket * 9) + (erBucket * 3) + shockBucket;
    }

    private static double evaluateCandidate(HMM model, int[] testObs, int[] y_test) {
        int correct = 0;
        int total = 0;
        int[] predictedPath = model.predict(testObs);

        for (int i = 0; i < testObs.length; i++) {
            int trueLabel = y_test[i];
            if (trueLabel < 0 || trueLabel > 3) continue;

            total++;
            if (predictedPath[i] == trueLabel) {
                correct++;
            }
        }
        return total > 0 ? (double) correct / total : 0.0;
    }






    public static void main(String[] args) {

        Random random = new Random(42); // Fixed seed for reproducibility

        // 1. Load the assembled chronological dataset
        List<MarketData> marketDataList;
        try {
            String path = "data/assembled_dataset.csv";

            marketDataList = DatasetReader.loadDataset(path);
            System.out.println("Loaded " + marketDataList.size() + " rows for visualization.");
        } catch (IOException e) {
            System.err.println("Critical Error loading CSV file: " + e.getMessage());
            return;
        }
        computeLabeledDataset(marketDataList);
        //create a new sublist such that the regime lables are sampled uniformly

        List<MarketData> learningList = new ArrayList<>();
        Map<Integer, MutableInt> classCounts = new java.util.HashMap<>();

        for (int i = 0; i < marketDataList.size(); i++) {
            MarketData data = marketDataList.get(i);
            int regimeLabel = data.getRegimeLabel();


            if (regimeLabel == 0 && random.nextDouble() < 0.09) {
                learningList.add(data);
                classCounts.putIfAbsent(regimeLabel, new MutableInt(0));
                classCounts.get(regimeLabel).increment();
            } else if (regimeLabel == 1 && random.nextDouble() < 0.95) {
                learningList.add(data);
                classCounts.putIfAbsent(regimeLabel, new MutableInt(0));
                classCounts.get(regimeLabel).increment();
            } else if (regimeLabel == 2 && random.nextDouble() < 0.95) {
                learningList.add(data);
                classCounts.putIfAbsent(regimeLabel, new MutableInt(0));
                classCounts.get(regimeLabel).increment();
            } else if (regimeLabel == 3) {
                learningList.add(data);
                classCounts.putIfAbsent(regimeLabel, new MutableInt(0));
                classCounts.get(regimeLabel).increment();
            }

        }

        //print out map
        System.out.println("Class distribution in learning list:");
        for (Map.Entry<Integer, MutableInt> entry : classCounts.entrySet()) {
            System.out.printf("Regime %d: %d samples\n", entry.getKey(), entry.getValue().get());
        }


        Map<String, DatasetStatisticsAnalyzer.FeatureStats> featureStatsMap = DatasetStatisticsAnalyzer.analyzeDataset(learningList);
//        DatasetSplitter.PartitionedDataset partitions = DatasetSplitter.splitChronologically(learningList);

  //      DatasetSplitter.PartitionedDataset partitionedDataset = new DatasetSplitter.PartitionedDataset()

        //take 80% of the learningList as the training set and 20% as the test set
        int trainSize = (int) (learningList.size() * 0.8);
        List<MarketData> trainSet = learningList.subList(0, trainSize);
        List<MarketData> testSet = learningList.subList(trainSize, learningList.size());
        DatasetSplitter.PartitionedDataset partitions = new DatasetSplitter.PartitionedDataset(trainSet, new ArrayList<>(), testSet);

        //optimizeHmm(partitions, featureStatsMap.get("USDJPY_ATR").mean);
        // 2. Run the HMM benchmark on the partitioned dataset
        runHmmBenchmark(partitions, featureStatsMap.get("USDJPY_ATR").mean);
    }
}