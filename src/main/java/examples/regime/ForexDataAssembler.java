package examples.regime;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class ForexDataAssembler {
    private final DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd HH:mm:ss");

    // Static bounds based on your requirements
    private final LocalDateTime startBound = LocalDateTime.parse("20171009 16:00:00", formatter);
    private final LocalDateTime endBound   = LocalDateTime.parse("20240508 21:00:00", formatter);

    // Synchronized storage: Time -> (Symbol -> Price)
    private final Map<LocalDateTime, Map<String, Double>> synchronizedData = new TreeMap<>();
    private final Map<String, Double> lastSeenPrices = new HashMap<>();

    /**
     * Loads a CSV file (e.g., USDJPY_4_min.csv) and filters for top-of-the-hour snapshots.
     */
    public void loadCSV(String filePath, String symbol) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            br.readLine(); // Skip header
            while ((line = br.readLine()) != null) {
                String[] v = line.split(",");
                LocalDateTime ldt = LocalDateTime.parse(v[0], formatter);

                // Filter: within specified range and exactly on the hour
                if (ldt.isBefore(startBound) || ldt.isAfter(endBound)) continue;
                if (ldt.getMinute() != 0) continue;

                double bidClose = Double.parseDouble(v[4]);
                double askClose = Double.parseDouble(v[9]);
                double mid = (bidClose + askClose) / 2.0;

                synchronizedData.computeIfAbsent(ldt, k -> new HashMap<>()).put(symbol, mid);
            }
        }
    }

    /**
     * Synchronizes multi-asset CSV data into hourly snapshots.
     * Calculates Point-in-Time features and Trajectory (Delta) features.
     */
    public List<MarketDataRaw> assemble(String mainPair) {
        List<MarketDataRaw> dataset = new ArrayList<>();
        List<LocalDateTime> timeline = new ArrayList<>(synchronizedData.keySet());

        // Global cache for Forward-Filling bond/oil prices
        Map<String, Double> lastSeenPrices = new HashMap<>();

        // We start at index 25 to ensure we have enough history to calculate
        // the current state AND the previous state for the Delta calculation.
        for (int i = 25; i < timeline.size(); i++) {
            LocalDateTime now = timeline.get(i);
            Map<String, Double> currentHourData = synchronizedData.get(now);

            // 1. Update Global Cache for Forward-Fill
            for (Map.Entry<String, Double> entry : currentHourData.entrySet()) {
                lastSeenPrices.put(entry.getKey(), entry.getValue());
            }

            // 2. Filter: Primary pair must be trading this hour
            if (!currentHourData.containsKey(mainPair)) continue;

            MarketDataRaw md = new MarketDataRaw();
            md.timestamp = now;

            // --- CURRENT STATE (T0) ---
            md.usdJpyAtr       = calculateNormalizedAtr(mainPair, i, timeline, 14);
            md.efficiencyRatio = calculateEfficiencyRatio(getRawPriceSequence(mainPair), i - 10, i);
            md.usdJpyChange    = calculateLogReturn(mainPair, i, timeline);

            // --- PREVIOUS STATE (T-1) for Delta calculation ---
            double prevAtr = calculateNormalizedAtr(mainPair, i - 1, timeline, 14);
            double prevEr  = calculateEfficiencyRatio(getRawPriceSequence(mainPair), i - 11, i - 1);

            // --- DELTA FEATURES (Momentum/Trajectory) ---
            // Positive = Expanding Volatility/Signal; Negative = Contracting
            md.atrDelta = md.usdJpyAtr - prevAtr;
            md.erDelta  = md.efficiencyRatio - prevEr;

            // --- MACRO LEVELS & SPREAD DELTA ---
            double ief = lastSeenPrices.getOrDefault("IEF", 0.0);
            double jgb = lastSeenPrices.getOrDefault("JGB", 0.0);
            md.usBondPrice = ief;
            md.jpBondPrice = jgb;
            md.oilPrice    = lastSeenPrices.getOrDefault("WTI", 0.0);

            // Yield Spread Delta (Trajectory of interest rate divergence)
            // Note: Using prices as proxies; a falling price difference implies a widening yield spread
            double currentSpread = ief - jgb;
            double prevIef = getVal("IEF", i - 1, timeline);
            double prevJgb = getVal("JGB", i - 1, timeline);
            double prevSpread = (prevIef != 0 && prevJgb != 0) ? (prevIef - prevJgb) : currentSpread;
            md.yieldSpreadDelta = currentSpread - prevSpread;

            // --- RELATIVE VOLATILITY CROSSES ---
            md.eurUsdAtr = calculateNormalizedAtr("EUR", i, timeline, 14);
            md.eurJpyAtr = calculateNormalizedAtr("EURJPY", i, timeline, 14);
            md.audJpyAtr = calculateNormalizedAtr("AUDJPY", i, timeline, 14);

            md.newsShock = 0; // Integration point for news shocks
            dataset.add(md);
        }
        return dataset;
    }



    /**
     * Retrieves value with historical forward-fill to prevent nulls during indicator calculation.
     */
    private double getVal(String sym, int idx, List<LocalDateTime> t) {
        if (idx < 0) return 0.0;
        Double val = synchronizedData.get(t.get(idx)).get(sym);
        if (val == null) {
            for (int prev = idx - 1; prev >= 0; prev--) {
                val = synchronizedData.get(t.get(prev)).get(sym);
                if (val != null) break;
            }
        }
        return (val != null) ? val : 0.0;
    }

    private double calculateNormalizedAtr(String sym, int idx, List<LocalDateTime> t, int period) {
        double currentPrice = getVal(sym, idx, t);
        if (currentPrice == 0) return 0.0;
        double sumTR = 0;
        for (int j = idx - period; j < idx; j++) {
            sumTR += Math.abs(getVal(sym, j + 1, t) - getVal(sym, j, t));
        }
        return (sumTR / period) / currentPrice;
    }

    public double calculateEfficiencyRatio(List<Double> prices, int start, int end) {
        if (start < 0 || end >= prices.size() || start >= end) return 0.0;
        double net = Math.abs(prices.get(end) - prices.get(start));
        double path = 0.0;
        for (int j = start + 1; j <= end; j++) {
            path += Math.abs(prices.get(j) - prices.get(j - 1));
        }
        return (path == 0) ? 0 : net / path;
    }

    private double calculateLogReturn(String sym, int idx, List<LocalDateTime> t) {
        double current = getVal(sym, idx, t);
        double prev = getVal(sym, idx - 1, t);
        return (prev == 0) ? 0 : Math.log(current / prev) * 100;
    }


    // Helper to extract prices for the RegimeLabeler
    public List<Double> getRawPriceSequence(String sym) {
        List<Double> prices = new ArrayList<>();
        for (LocalDateTime time : synchronizedData.keySet()) {
            prices.add(synchronizedData.get(time).getOrDefault(sym, 0.0));
        }
        return prices;
    }



    public static class MarketDataRaw {
        public LocalDateTime timestamp;
        public double usdJpyChange, usdJpyAtr, efficiencyRatio;
        public double usBondPrice, jpBondPrice, oilPrice;
        public double eurUsdAtr, eurJpyAtr, audJpyAtr;

        // Delta Features (Directional Lags)
        public double atrDelta; // ATR(t) - ATR(t-1)
        public double erDelta;  // ER(t) - ER(t-1)
        public double yieldSpreadDelta; // (US-JP Spread)_t - (US-JP Spread)_t-1

        public int newsShock;

        @Override
        public String toString() {
            return String.format("Time: %s | PriceChange: %.4f | ATR: %.4f | ER: %.4f | USBond: %.2f | JPBond: %.2f | Oil: %.2f | EURUSD_ATR: %.4f | EURJPY_ATR: %.4f | AUDJPY_ATR: %.4f | NewsShock: %d",
                    timestamp, usdJpyChange, usdJpyAtr, efficiencyRatio, usBondPrice, jpBondPrice, oilPrice,
                    eurUsdAtr, eurJpyAtr, audJpyAtr, newsShock);
        }
    }
}