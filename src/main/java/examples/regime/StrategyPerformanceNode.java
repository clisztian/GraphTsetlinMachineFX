package examples.regime;

public class StrategyPerformanceNode {
    private final int lookbackWindow; // e.g., 6 or 12 hours
    private final int perfWindow;     // e.g., 24 or 48 hours for the Sortino/Sharpe calculation
    private final double[] returnHistory;
    private final double[] strategyReturnHistory;
    private int writeIndex = 0;
    private int totalElements = 0;

    public StrategyPerformanceNode(int lookbackWindow, int perfWindow) {
        this.lookbackWindow = lookbackWindow;
        this.perfWindow = perfWindow;
        this.returnHistory = new double[perfWindow * 2]; // Give ample buffer space
        this.strategyReturnHistory = new double[perfWindow];
    }

    /**
     * Updates the node with the latest hourly log return and outputs the current rolling performance.
     * @param currentLogReturn ln(Price_t / Price_t-1)
     * @return Point-in-time Sortino Ratio or raw performance score
     */
    public double updateAndComputePerformance(double currentLogReturn) {
        // 1. Store the raw log return
        returnHistory[writeIndex % returnHistory.length] = currentLogReturn;

        if (totalElements < lookbackWindow) {
            totalElements++;
            writeIndex++;
            return 0.0; // Not enough data to compute momentum yet
        }

        // 2. Compute the momentum sign over the lookback window (t-1 context)
        double cumulativeLookbackReturn = 0.0;
        for (int i = 1; i <= lookbackWindow; i++) {
            int idx = (writeIndex - i + returnHistory.length) % returnHistory.length;
            cumulativeLookbackReturn += returnHistory[idx];
        }

        // Position decided at t-1 dictates returns at t
        int position = (cumulativeLookbackReturn >= 0) ? 1 : -1;
        double currentStrategyReturn = position * currentLogReturn;

        // 3. Store the strategy return in a rolling window
        int stratIdx = writeIndex % perfWindow;
        strategyReturnHistory[stratIdx] = currentStrategyReturn;

        writeIndex++;
        if (totalElements < perfWindow) {
            totalElements++;
            return 0.0; // Not enough history to calculate rolling Sortino metric
        }

        return calculateRollingSortino();
    }

    private double calculateRollingSortino() {
        double sumReturns = 0.0;
        double downsideVariance = 0.0;

        for (double r : strategyReturnHistory) {
            sumReturns += r;
            if (r < 0) {
                downsideVariance += (r * r);
            }
        }

        double meanReturn = sumReturns / perfWindow;
        double downsideDeviation = Math.sqrt(downsideVariance / perfWindow);

        // Avoid division by zero if there is no downside risk yet
        if (downsideDeviation == 0.0) {
            return meanReturn > 0 ? 1.0 : -1.0;
        }

        return meanReturn / downsideDeviation;
    }
}