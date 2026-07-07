package examples.regime.io;

import examples.regime.dto.MarketData;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class MarketDataExporter {

    private static final String CSV_HEADER =
            "Timestamp,USDJPY_PRICE,USDJPY_ATR,USDJPY_ER,EURUSD_ATR,AUDJPY_ATR,EURJPY_ATR," +
                    "US_BOND_LEVEL,JP_BOND_LEVEL,WTI_OIL,YIELD_SPREAD_DELTA,ATR_DELTA,ER_DELTA,LABEL";

    /**
     * Exports the internal structural timeline directly to a standardized CSV archive.
     * * @param dataset  The full in-memory list of populated MarketData rows
     * @param filePath The local destination file path (e.g., "data/processed_regimes.csv")
     */
    public static void exportToCsv(List<MarketData> dataset, String filePath) {
        if (dataset == null || dataset.isEmpty()) {
            System.err.println("[EXPORTER ERROR] Dataset is empty or null. Export aborted.");
            return;
        }

        System.out.printf("Initiating file export sequence for %d time-series rows...\n", dataset.size());
        long startTime = System.currentTimeMillis();

        // Use standard UTF-8 explicit encoding inside a try-with-resources to guarantee safe file closures
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath, StandardCharsets.UTF_8))) {

            // 1. Write target structural file header
            writer.write(CSV_HEADER);
            writer.newLine();

            // 2. Reuse a single StringBuilder capacity framework to limit runtime heap memory churn
            StringBuilder sb = new StringBuilder(256);

            for (MarketData row : dataset) {
                // Skip edge tracking rows that failed alignment initialization or are unlabelled (-1)
                if (row.getRegimeLabel() == -1) {
                    continue;
                }

                sb.setLength(0); // High-speed clear buffer length index

                // Append matching point-in-time sequential primitives exactly mirroring header keys
                sb.append(row.getTimestamp()).append(",")
                        .append(row.getUsdjpyPrice()).append(",")
                        .append(row.getUsdjpyAtr()).append(",")
                        .append(row.getUsdjpyEr()).append(",")
                        .append(row.getEurusdAtr()).append(",")
                        .append(row.getAudjpyAtr()).append(",")
                        .append(row.getEurjpyAtr()).append(",")
                        .append(row.getUsBondLevel()).append(",")
                        .append(row.getJpBondLevel()).append(",")
                        .append(row.getWtiOil()).append(",")
                        .append(row.getYieldSpreadDelta()).append(",")
                        .append(row.getAtrDelta()).append(",")
                        .append(row.getErDelta()).append(",")
                        .append(row.getRegimeLabel());

                writer.write(sb.toString());
                writer.newLine();
            }

            // Force physical flush block to mechanical sector tracks
            writer.flush();

            long endTime = System.currentTimeMillis();
            System.out.printf("[SUCCESS] Dataset safely written to '%s' in %d ms.\n", filePath, (endTime - startTime));

        } catch (IOException e) {
            System.err.println("[CRITICAL ERROR] Failed writing tracking loop arrays out to disk matrix: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
