package util;

public class Utility {

    /**
     * Computes the asymmetric financial risk score for a 4x4 regime confusion matrix.
     * Lower scores indicate safer, more profitable models for your trading strategy.
     * * @param confusionMatrix A 4x4 integer array where rows are actuals and columns are predictions.
     * @return The normalized risk score per trading hour.
     */
    public static double calculateTradingRiskScore(int[][] confusionMatrix) {
        // 1. Define your asymmetric cost matrix based on strategy constraints
        // Rows = Actual Regime (0,1,2,3) | Columns = Predicted Regime (0,1,2,3)
        final double[][] costMatrix = {
                {0.0, 1.0,  3.0,  1.0},  // Actual 0 (Sideways): Predicting trend is safe (vol filter catches it)
                {4.0, 0.0,  4.0,  2.0},  // Actual 1 (Trending): Missing a trend loses potential profit
                {8.0, 10.0, 0.0, 10.0},  // Actual 2 (Choppy): PREDICTING TREND IS CATASTROPHIC (Whipsaw penalty)
                {5.0, 2.0,  4.0,  0.0}   // Actual 3 (Shock): Missing a massive shock trend is costly
        };

        double totalRisk = 0.0;
        long totalSamples = 0;

        // 2. Element-wise multiplication of the confusion matrix by your operational costs
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int cellCount = confusionMatrix[i][j];
                totalRisk += cellCount * costMatrix[i][j];
                totalSamples += cellCount;
            }
        }

        // 3. Prevent division by zero if an empty matrix is passed
        if (totalSamples == 0) {
            return 0.0;
        }

        // Return the normalized risk per trading hour
        return totalRisk / totalSamples;
    }

}
