package examples.regime.macro;


import examples.regime.dto.MarketData;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class MacroShockProcessor {

    // Model DTO representing the inbound API payload structures
    public static class EconomicReleaseEvent {
        public LocalDateTime releaseTime;
        public String eventName; // e.g., "US Core CPI YoY" or "US Inflation Rate MoM"
        public String currency;  // "USD" or "JPY"
        public double actual;
        public double consensus;

        public EconomicReleaseEvent(LocalDateTime releaseTime, String eventName, String currency, double actual, double consensus) {
            this.releaseTime = releaseTime;
            this.eventName = eventName;
            this.currency = currency;
            this.actual = actual;
            this.consensus = consensus;
        }
    }

    private static final int SHOCK_PERSISTENCE_HOURS = 72;

    // Quantitative thresholds adjusted for annualized % representations (e.g., 3.1% actual vs 2.9% consensus)
    private static final double MODERATE_SURPRISE_LIMIT = 0.10; // 10 bps surprise deviation
    private static final double SEVERE_SURPRISE_LIMIT = 0.25;   // 25+ bps surprise deviation

    /**
     * Map historical event data arrays directly onto your prepared hourly continuous dataset.
     * * @param hourlyDataset The continuous chronological List of MarketData loaded from your CSV
     * @param rawEvents The collection of past CPI/Inflation shock prints fetched from your API
     */
    public static void computeAndInjectCPIShocks(List<MarketData> hourlyDataset, List<EconomicReleaseEvent> rawEvents) {

        // Loop through each hourly row in your dataset matrix
        for (int t = 0; t < hourlyDataset.size(); t++) {
            MarketData currentHour = hourlyDataset.get(t);
            LocalDateTime currentHourTime = currentHour.getTimestamp();

            int highestActiveShockValue = 0;

            // Scan the event timeline to see if any macro shock applies to the current 72-hour window
            for (EconomicReleaseEvent event : rawEvents) {
                // If the event happened in the past relative to row 't', but within a 72-hour decay boundary
                if (!event.releaseTime.isAfter(currentHourTime) &&
                        event.releaseTime.plusHours(SHOCK_PERSISTENCE_HOURS).isAfter(currentHourTime)) {

                    // 1. Calculate absolute surprise metric
                    double rawSurprise = Math.abs(event.actual - event.consensus);

                    // 2. Classify shock intensity layer
                    int shockIntensity = 0;
                    if (rawSurprise >= SEVERE_SURPRISE_LIMIT) {
                        shockIntensity = 2; // Black Swan / Regime-Altering repricing shock
                    } else if (rawSurprise >= MODERATE_SURPRISE_LIMIT) {
                        shockIntensity = 1; // Standard statistical trend acceleration driver
                    }

                    // Keep the maximum observed shock value if multiple macro events overlap in the 72h window
                    if (shockIntensity > highestActiveShockValue) {
                        highestActiveShockValue = shockIntensity;
                    }
                }
            }

            // 3. Persist the calculated property token directly inside your node properties
            currentHour.setUsdCpiShock(highestActiveShockValue);
        }

        System.out.println("CPI Macro Shock features successfully mapped across the 10-year continuous timeline.");
    }

    /**
     * Mock execution block simulating how this integrates with your real processing streams
     */
    public static void main(String[] args) {
        // Mock dataset containing a snippet of chronological hours
        List<MarketData> mockDataset = new ArrayList<>();
        LocalDateTime startTime = LocalDateTime.of(2017, 10, 10, 12, 0);
        for (int i = 0; i < 100; i++) {
            MarketData data = new MarketData();
            data.setTimestamp(startTime.plusHours(i));
            mockDataset.add(data);
        }

        // Mock an API payload event: Core CPI prints 2.4% vs 2.1% expected (A massive 30bps surprise)
        List<EconomicReleaseEvent> mockApiPayload = new ArrayList<>();
        mockApiPayload.add(new EconomicReleaseEvent(
                LocalDateTime.of(2017, 10, 11, 8, 0), // Occurs at hour index 20
                "US Core CPI YoY", "USD", 2.4, 2.1
        ));

        // Process the injection
        computeAndInjectCPIShocks(mockDataset, mockApiPayload);

        // Verification Check: Print row properties right after the shock occurs vs 73 hours later
        System.out.println("Hour 21 Shock Token (1 hour post-release): " + mockDataset.get(21).getUsdCpiShock()); // Returns 2
        System.out.println("Hour 95 Shock Token (75 hours post-release): " + mockDataset.get(95).getUsdCpiShock()); // Returns 0 (Decayed away)
    }
}