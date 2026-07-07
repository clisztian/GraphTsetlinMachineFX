package examples.regime;

import examples.regime.dto.MarketData;
import util.MutableInt;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RegimeLabeler {

    // Define system-wide quantitative thresholds
    private static final double VOLATILITY_THRESHOLD = 0.08; // e.g., 8 basis points (0.08%)
    private static final double EFFICIENCY_THRESHOLD = 0.35;   // ER boundary between trend and noise
    private static final int LOOKAHEAD_WINDOW = 72;            // 3 days of hourly data
    private static final int ER_PERIOD = 20;                  // 10-hour window for Kaufman ER

    public static int computeLookAheadLabel(List<MarketData> dataset, int targetIndex) {

        if (targetIndex < ER_PERIOD || targetIndex + LOOKAHEAD_WINDOW >= dataset.size()) {
            return -1; // Insufficient history or look-ahead data; skip this sample cleanly
        }

        int[] regimeCounts = new int[4];

        // Iterate through each hour of the 72-hour look-ahead window
        for (int h = 1; h <= LOOKAHEAD_WINDOW; h++) {
            int evalHourIndex = targetIndex + h;

            // --- 1. Compute Instantaneous Volatility ---
            double hourlyLogReturn = dataset.get(evalHourIndex).getUsdjpyPrice();
            boolean isHighVol = Math.abs(hourlyLogReturn) >= VOLATILITY_THRESHOLD;

            // --- 2. Compute Instantaneous Efficiency Ratio (ER) ---
            // Because targetIndex >= ER_PERIOD, erStartIndex will always be >= 1 here.
            int erStartIndex = evalHourIndex - ER_PERIOD;
            double efficiencyRatio = calculateWindowER(dataset, erStartIndex, evalHourIndex);
            boolean isHighEfficiency = efficiencyRatio >= EFFICIENCY_THRESHOLD;


            int instantaneousRegime;
            if (!isHighVol && !isHighEfficiency) {
                instantaneousRegime = 0; // Stagnant
            } else if (!isHighVol && isHighEfficiency) {
                instantaneousRegime = 1; // Steady Grind
            } else if (isHighVol && !isHighEfficiency) {
                instantaneousRegime = 2; // Choppy / Mean-Reverting
            } else {
                instantaneousRegime = 3; // Volatile Trend / Directional Explosion
            }



            regimeCounts[instantaneousRegime]++;
        }

        // --- 4. Compute Majority Vote (Mode) ---
        int majorityRegime = 0;
        int maxCount = -1;
        for (int i = 0; i < regimeCounts.length; i++) {
            if (regimeCounts[i] > maxCount) {
                maxCount = regimeCounts[i];
                majorityRegime = i;
            }
        }

        return majorityRegime;
    }

    /**
     * Computes the point-in-time Kaufman Efficiency Ratio from underlying log returns.
     */
    private static double calculateWindowER(List<MarketData> dataset, int startIdx, int endIdx) {
        double netLogReturnSum = 0.0;
        double absoluteLogReturnSum = 0.0;

        for (int i = startIdx + 1; i <= endIdx; i++) {
            double logReturn = dataset.get(i).getUsdjpyPrice();
            netLogReturnSum += logReturn;
            absoluteLogReturnSum += Math.abs(logReturn);
        }

        // Net displacement = |P_end - P_start| represented in log space as |e^(sum of returns)|
        // For a normalized proxy close to 1.0, this maps cleanly to the absolute net log return sum
        double netDisplacement = Math.abs(netLogReturnSum);

        if (absoluteLogReturnSum == 0.0) {
            return 0.0;
        }

        double er = netDisplacement / absoluteLogReturnSum;
        return Math.min(1.0, Math.max(0.0, er)); // Bounded strictly between 0 and 1
    }


    public static void computeLabeledDataset(List<MarketData> marketHistory) {

        //count each regime label using a hash map for debugging purposes
        Map<Integer, MutableInt> regimeCounts = new HashMap<>();

        for (int t = 0; t < marketHistory.size(); t++) {
            int targetLabel = RegimeLabeler.computeLookAheadLabel(marketHistory, t);

            if (targetLabel == -1) {
                if (t < 24) continue;
                else break;
            }
            marketHistory.get(t).setRegimeLabel(targetLabel);
            regimeCounts.computeIfAbsent(targetLabel, k -> new MutableInt()).increment();
        }

        // Print out the distribution of regime labels for sanity check
        System.out.println("Regime Label Distribution:");
        for (Map.Entry<Integer, MutableInt> entry : regimeCounts.entrySet()) {
            System.out.printf("Regime %d: %d samples%n", entry.getKey(), entry.getValue().get());
        }
    }


}