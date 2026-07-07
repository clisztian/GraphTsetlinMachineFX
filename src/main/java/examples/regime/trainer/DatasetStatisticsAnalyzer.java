package examples.regime.trainer;

import examples.regime.dto.MarketData;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DatasetStatisticsAnalyzer {

    /**
     * Container class to store calculated statistics for a single feature.
     */
    public static class FeatureStats {
        public double min = Double.MAX_VALUE;
        public double max = -Double.MAX_VALUE;
        public double sum = 0.0;
        public double mean = 0.0;

        public void update(double value) {
            if (value < min) min = value;
            if (value > max) max = value;
            sum += value;
        }

        public void finalizeStats(int totalSamples) {
            if (totalSamples > 0) {
                this.mean = this.sum / totalSamples;
            }
        }

        @Override
        public String toString() {
            return String.format("Min: %12.6f | Max: %12.6f | Mean: %12.6f", min, max, mean);
        }
    }

    /**
     * Iterates through the processed dataset and calculates min, max, and mean
     * for all hypervector input features.
     * * @param dataset The list of processed MarketData objects (ideally your training set partition)
     * @return A map containing the descriptive statistics keyed by feature name
     */
    public static Map<String, FeatureStats> analyzeDataset(List<MarketData> dataset) {
        Map<String, FeatureStats> statsMap = new HashMap<>();

        // Initialize tracking containers for every core graph node feature
        String[] features = {
                "USDJPY_PRICE", "USDJPY_ATR", "USDJPY_ER",
                "US_BOND_LEVEL", "JP_BOND_LEVEL", "WTI_OIL",
                "ATR_DELTA", "ER_DELTA", "YIELD_SPREAD_DELTA",
                "EURUSD_ATR", "AUDJPY_ATR", "EURJPY_ATR"
        };

        for (String feature : features) {
            statsMap.put(feature, new FeatureStats());
        }

        int validCount = 0;

        // 1. Single chronological pass to collect boundaries and sums
        for (MarketData row : dataset) {
            // Optional Safety: Skip row entirely if it was blocked during labeling initialization
            if (row.getRegimeLabel() == -1) {
                continue;
            }
            validCount++;

            statsMap.get("USDJPY_PRICE").update(row.getUsdjpyPrice());
            statsMap.get("USDJPY_ATR").update(row.getUsdjpyAtr());
            statsMap.get("USDJPY_ER").update(row.getUsdjpyEr());
            statsMap.get("US_BOND_LEVEL").update(row.getUsBondLevel());
            statsMap.get("JP_BOND_LEVEL").update(row.getJpBondLevel());
            statsMap.get("WTI_OIL").update(row.getWtiOil());
            statsMap.get("ATR_DELTA").update(row.getAtrDelta());
            statsMap.get("ER_DELTA").update(row.getErDelta());
            statsMap.get("YIELD_SPREAD_DELTA").update(row.getYieldSpreadDelta());
            statsMap.get("EURUSD_ATR").update(row.getEurusdAtr());
            statsMap.get("AUDJPY_ATR").update(row.getAudjpyAtr());
            statsMap.get("EURJPY_ATR").update(row.getEurjpyAtr());
        }

        // 2. Compute means across all feature containers
        System.out.println("======================================================================");
        System.out.printf("   DESCRIPTIVE DATASET METRICS ANALYSIS (Valid Samples: %d)\n", validCount);
        System.out.println("======================================================================");

        for (String feature : features) {
            FeatureStats stats = statsMap.get(feature);
            stats.finalizeStats(validCount);
            System.out.printf("%-20s -> %s\n", feature, stats.toString());
        }
        System.out.println("======================================================================");

        return statsMap;
    }
}
