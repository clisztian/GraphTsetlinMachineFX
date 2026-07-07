package examples.regime.trainer;

import examples.regime.dto.MarketData;
import examples.regime.io.DatasetReader;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static examples.regime.RegimeLabeler.computeLabeledDataset;
import static examples.regime.trainer.TrainingWeightCalibrator.calculateClassUpdateProbabilities;

public class DatasetSplitter {

    private static final int LOOKAHEAD_WINDOW = 72; // Hours to purge
    private static final int EMBARGO_WINDOW = 24;   // Hours to embargo

    public static class PartitionedDataset {
        public final List<MarketData> trainSet;
        public final List<MarketData> valSet;
        public final List<MarketData> testSet;

        public PartitionedDataset(List<MarketData> trainSet, List<MarketData> valSet, List<MarketData> testSet) {
            this.trainSet = trainSet;
            this.valSet = valSet;
            this.testSet = testSet;
        }
    }

    /**
     * Chronologically splits the dataset into Train, Validation, and Test sets
     * while completely purging overlapping look-ahead labels and embargoing leakage.
     */
    public static PartitionedDataset splitChronologically(List<MarketData> completeDataset) {
        int totalSize = completeDataset.size();

        // 1. Calculate raw chronological index boundaries (60% / 20% / 20%)
        int rawTrainEnd = (int) (totalSize * 0.60);
        int rawValEnd = (int) (totalSize * 0.80);


        int purgedTrainEnd = rawTrainEnd - LOOKAHEAD_WINDOW;
        List<MarketData> trainSet = new ArrayList<>(completeDataset.subList(500, purgedTrainEnd));


        int embargoedValStart = rawTrainEnd + EMBARGO_WINDOW;
        int purgedValEnd = rawValEnd - LOOKAHEAD_WINDOW;
        List<MarketData> valSet = new ArrayList<>(completeDataset.subList(embargoedValStart, purgedValEnd));


        int embargoedTestStart = rawValEnd + EMBARGO_WINDOW;

        int cleanTestEnd = totalSize;
        while (cleanTestEnd > embargoedTestStart && completeDataset.get(cleanTestEnd - 1).getRegimeLabel() == -1) {
            cleanTestEnd--; // Skip unlabeled samples at the absolute trailing edge of history
        }
        List<MarketData> testSet = new ArrayList<>(completeDataset.subList(embargoedTestStart, cleanTestEnd));

        return new PartitionedDataset(trainSet, valSet, testSet);
    }

    // Quick structural integrity verification execution block
    public static void main(String[] args) {
        try {

            List<MarketData> completeDataset = DatasetReader.loadDataset("data/assembled_dataset.csv");
            //add the label
            computeLabeledDataset(completeDataset);

            Map<String, DatasetStatisticsAnalyzer.FeatureStats> featureStatsMap = DatasetStatisticsAnalyzer.analyzeDataset(completeDataset);

            calculateClassUpdateProbabilities(completeDataset);


            System.out.println("Total Assembled Samples Available: " + completeDataset.size());

            PartitionedDataset partitions = splitChronologically(completeDataset);

            System.out.println("\n--- Leakage-Protected Partition Map ---");
            System.out.printf("Train Set Matrix Size:       %d samples  (Starts: %s -> Ends: %s)\n",
                    partitions.trainSet.size(), partitions.trainSet.get(0).getTimestamp(), partitions.trainSet.get(partitions.trainSet.size()-1).getTimestamp());
            System.out.printf("Validation Set Matrix Size:  %d samples  (Starts: %s -> Ends: %s)\n",
                    partitions.valSet.size(), partitions.valSet.get(0).getTimestamp(), partitions.valSet.get(partitions.valSet.size()-1).getTimestamp());
            System.out.printf("Test Set Matrix Size:        %d samples  (Starts: %s -> Ends: %s)\n",
                    partitions.testSet.size(), partitions.testSet.get(0).getTimestamp(), partitions.testSet.get(partitions.testSet.size()-1).getTimestamp());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}