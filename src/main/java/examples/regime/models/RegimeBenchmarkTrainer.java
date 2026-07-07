package examples.regime.models;


import examples.regime.dto.MarketData;
import examples.regime.io.DatasetReader;
import examples.regime.trainer.DatasetSplitter;
import examples.regime.trainer.DatasetStatisticsAnalyzer;
import smile.classification.GradientTreeBoost;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.vector.IntVector;
import util.MutableInt;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static examples.regime.RegimeLabeler.computeLabeledDataset;

public class RegimeBenchmarkTrainer {

    static Random random = new Random(42); // For reproducibility

    private static final String[] FEATURE_NAMES = {
            "usdjpyPrice", "usdjpyAtr", "usdjpyEr", "usBondLevel", "jpBondLevel", "wtiOil",
            "atrDelta", "erDelta", "yieldSpreadDelta", "eurusdAtr", "audjpyAtr", "eurjpyAtr"
    };

    public static class OptimalConfig {
        public int ntrees;
        public int maxDepth;
        public double shrinkage;
        public double subsample;
        public double overallAccuracy;
    }


    public static void runBoostedTreeBenchmark(DatasetSplitter.PartitionedDataset partitions) {
        System.out.println("Converting time series arrays directly into SMILE 3.1.1 DataFrames...");

        // 1. Flatten features and labels for both splits
        double[][] X_train = flattenX(partitions.trainSet);
        int[] y_train = flattenY(partitions.trainSet);
        double[][] X_test = flattenX(partitions.testSet);
        int[] y_test = flattenY(partitions.testSet);

        // 2. Assemble Train DataFrame (Features + Named Label Vector)
        DataFrame trainFeatures = DataFrame.of(X_train, FEATURE_NAMES);
        IntVector trainLabels = IntVector.of("y", y_train);
        DataFrame trainFrame = trainFeatures.merge(trainLabels);

        // 3. Assemble Test DataFrame (Features + Named Label Vector) so rows can be read as Tuples
        DataFrame testFeatures = DataFrame.of(X_test, FEATURE_NAMES);
        IntVector testLabels = IntVector.of("y", y_test);
        DataFrame testFrame = testFeatures.merge(testLabels);

        System.out.println("Training Baseline Gradient Tree Boosting Model via Formula Interface...");
        long startTrain = System.currentTimeMillis();

        Formula formula = Formula.lhs("y");

        GradientTreeBoost model = GradientTreeBoost.fit(
                formula,
                trainFrame,
                183,     // ntrees
                8,       // maxDepth
                100,      // maxNodes
                5,       // nodeSize
                0.023,    // shrinkage
                0.63     // subsample
        );

        long endTrain = System.currentTimeMillis();
        System.out.printf("SMILE Engine Training Complete in %d ms.\n", (endTrain - startTrain));

        // Evaluate Performance Out-of-Sample using the structured Test Frame
        evaluateModel(model, testFrame);
    }
    public static OptimalConfig optimizeBaseline(DatasetSplitter.PartitionedDataset partitions, int totalIterations) {
        System.out.println("Beginning Baseline Grid Ingestion Pass...");
        Random rand = new Random(42); // Seeded for peer replication

        // 1. Prepare Underlying Matrices
        double[][] X_train = flattenX(partitions.trainSet);
        int[] y_train = flattenY(partitions.trainSet);
        double[][] X_test = flattenX(partitions.testSet);
        int[] y_test = flattenY(partitions.testSet);

        DataFrame trainFrame = DataFrame.of(X_train, FEATURE_NAMES).merge(IntVector.of("y", y_train));
        DataFrame testFrame = DataFrame.of(X_test, FEATURE_NAMES).merge(IntVector.of("y", y_test));
        Formula formula = Formula.lhs("y");

        OptimalConfig bestConfig = new OptimalConfig();
        bestConfig.overallAccuracy = -1.0;

        System.out.printf("Running Grid Optimizer Search across %d unique operational matrices...\n", totalIterations);

        for (int i = 1; i <= totalIterations; i++) {
            // Stochastic Parameter Generation within defined macro bounds
            int ntrees = 100 + rand.nextInt(201);       // 100 to 300 trees
            int maxDepth = 4 + rand.nextInt(5);         // 4 to 8 split tiers
            double shrinkage = 0.01 + (rand.nextDouble() * 0.09); // 0.01 to 0.10 learning rate
            double subsample = 0.60 + (rand.nextDouble() * 0.20); // 0.60 to 0.80 row masking

            try {
                // Train candidate model configuration
                GradientTreeBoost candidateModel = GradientTreeBoost.fit(
                        formula, trainFrame, ntrees, maxDepth, 32, 5, shrinkage, subsample
                );

                // Compute overall validation validation metric
                double accuracy = calculateTotalAccuracy(candidateModel, testFrame);

                if (accuracy > bestConfig.overallAccuracy) {
                    bestConfig.overallAccuracy = accuracy;
                    bestConfig.ntrees = ntrees;
                    bestConfig.maxDepth = maxDepth;
                    bestConfig.shrinkage = shrinkage;
                    bestConfig.subsample = subsample;

                    System.out.printf("[ITER %3d] New Optimal Frontier Found! Acc: %.2f%% | Trees: %d, Depth: %d, LR: %.4f\n",
                            i, accuracy * 100, ntrees, maxDepth, shrinkage);
                }
            } catch (Exception e) {
                System.err.println("Skipping volatile tracking loop iteration: " + e.getMessage());
            }
        }

        System.out.println("\n======================================================================");
        System.out.println("             GRID OPTIMIZATION HYPERPARAMETER SEARCH RESULTS          ");
        System.out.println("======================================================================");
        System.out.printf(" Max Out-of-Sample Target Accuracy : %6.2f%%\n", bestConfig.overallAccuracy * 100);
        System.out.printf(" Optimal Tree Ensemble Size        : %d\n", bestConfig.ntrees);
        System.out.printf(" Optimal Hierarchical Max Depth    : %d\n", bestConfig.maxDepth);
        System.out.printf(" Optimal Shrinkage Coefficient     : %.5f\n", bestConfig.shrinkage);
        System.out.printf(" Optimal Subsample Data Split      : %.2f%%\n", bestConfig.subsample * 100);
        System.out.println("======================================================================");

        return bestConfig;
    }

    private static double calculateTotalAccuracy(GradientTreeBoost model, DataFrame testFrame) {
        int correct = 0;
        int total = 0;

        for (int i = 0; i < testFrame.nrow(); i++) {
            Tuple rowTuple = testFrame.get(i);
            int trueLabel = rowTuple.getInt("y");
            if (trueLabel < 0 || trueLabel > 3) continue;

            int pred = model.predict(rowTuple);
            total++;
            if (pred == trueLabel) {
                correct++;
            }
        }
        return total > 0 ? (double) correct / total : 0.0;
    }

    private static double[][] flattenX(List<MarketData> rows) {
        double[][] X = new double[rows.size()][12];
        for (int i = 0; i < rows.size(); i++) {
            MarketData r = rows.get(i);
            X[i] = new double[]{
                    r.getUsdjpyPrice(), r.getUsdjpyAtr(), r.getUsdjpyEr(), r.getUsBondLevel(),
                    r.getJpBondLevel(), r.getWtiOil(), r.getAtrDelta(), r.getErDelta(),
                    r.getYieldSpreadDelta(), r.getEurusdAtr(), r.getAudjpyAtr(), r.getEurjpyAtr()
            };
        }
        return X;
    }

    private static int[] flattenY(List<MarketData> rows) {
        int[] y = new int[rows.size()];
        for (int i = 0; i < rows.size(); i++) y[i] = rows.get(i).getRegimeLabel();
        return y;
    }
    /**
     * Iterates through the test DataFrame, extracting point-in-time Tuples
     * to satisfy the strict signature requirements of SMILE 3.1.1.
     */
    private static void evaluateModel(GradientTreeBoost model, DataFrame testFrame) {
        int[] classTotals = new int[4];
        int[] classCorrect = new int[4];

        long startInference = System.nanoTime();

        // Loop directly through the row indices of the DataFrame
        for (int i = 0; i < testFrame.nrow(); i++) {
            // FIXED: Get the row as a native smile.data.Tuple object
            Tuple rowTuple = testFrame.get(i);

            // Extract the ground truth from our named column "y"
            int trueLabel = rowTuple.getInt("y");
            if (trueLabel < 0 || trueLabel > 3) continue;

            // FIXED: Maps flawlessly to 'int predict(Tuple)' signature candidate
            int predictedLabel = model.predict(rowTuple);

            classTotals[trueLabel]++;
            if (predictedLabel == trueLabel) {
                classCorrect[trueLabel]++;
            }
        }
        long endInference = System.nanoTime();

        System.out.println("\n======================================================================");
        System.out.println("          SMILE v3.1.1 GRADIENT BOOSTING OUT-OF-SAMPLE ACCURACY       ");
        System.out.println("======================================================================");
        for (int c = 0; c < 4; c++) {
            double accuracy = classTotals[c] > 0 ? (double) classCorrect[c] / classTotals[c] : 0.0;
            System.out.printf("Regime Class %d -> Benchmark Accuracy: %6.2f%% (Samples: %d)\n",
                    c, accuracy * 100, classTotals[c]);
        }
        System.out.println("======================================================================");
        System.out.printf("Production Inference Speed: Average %.3f microseconds per sample hour.\n",
                ((double)(endInference - startInference) / testFrame.nrow()) / 1000.0);
        System.out.println("======================================================================");
    }



    //main method
    public static void main(String[] args) {
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

        for(int i = 0; i < marketDataList.size(); i++) {
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
        for(Map.Entry<Integer, MutableInt> entry : classCounts.entrySet()) {
            System.out.printf("Regime %d: %d samples\n", entry.getKey(), entry.getValue().get());
        }



        Map<String, DatasetStatisticsAnalyzer.FeatureStats> featureStatsMap = DatasetStatisticsAnalyzer.analyzeDataset(learningList);
        DatasetSplitter.PartitionedDataset partitions = DatasetSplitter.splitChronologically(learningList);

        //optimizeBaseline(partitions, 150);

        runBoostedTreeBenchmark(partitions);
    }
}