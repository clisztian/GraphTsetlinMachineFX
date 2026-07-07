package examples.regime.io;

import examples.regime.dto.MarketData;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DatasetReader {


    private static final DateTimeFormatter ISO_FORMATTER = DateTimeFormatter.ISO_LOCAL_DATE_TIME;

    public static List<MarketData> loadDataset(String filePath) throws IOException {
        List<MarketData> dataset = new ArrayList<>();

        String dataPath = DatasetReader.class.getClassLoader().getResource(filePath).getPath();

        try (BufferedReader br = new BufferedReader(new FileReader(dataPath))) {
            String headerLine = br.readLine();
            if (headerLine == null) {
                throw new IllegalArgumentException("The dataset file is empty: " + filePath);
            }

            // Split headers and thoroughly trim them to eliminate hidden layout gaps (e.g., " ATR_DELTA")
            String[] headers = headerLine.split(",");
            Map<String, Integer> colMap = new HashMap<>();
            for (int i = 0; i < headers.length; i++) {
                colMap.put(headers[i].trim().toUpperCase(), i);
            }

            String line;
            int lineNumber = 1;
            while ((line = br.readLine()) != null) {
                lineNumber++;
                String[] tokens = line.split(",", -1);

                try {
                    MarketData row = new MarketData();

                    // Parse Timestamp (e.g., 2017-10-10T17:00)
                    if (colMap.containsKey("TIMESTAMP")) {
                        row.setTimestamp(LocalDateTime.parse(tokens[colMap.get("TIMESTAMP")].trim(), ISO_FORMATTER));
                    }

                    // Levels
                    row.setUsdjpyPrice(getSafeDouble(tokens, colMap, "USDJPY_PRICE"));
                    row.setUsdjpyAtr(getSafeDouble(tokens, colMap, "USDJPY_ATR"));
                    row.setUsdjpyEr(getSafeDouble(tokens, colMap, "USDJPY_ER"));
                    row.setUsBondLevel(getSafeDouble(tokens, colMap, "US_BOND_LEVEL"));
                    row.setJpBondLevel(getSafeDouble(tokens, colMap, "JP_BOND_LEVEL"));
                    row.setWtiOil(getSafeDouble(tokens, colMap, "WTI_OIL"));

                    // Trajectories / Deltas
                    row.setAtrDelta(getSafeDouble(tokens, colMap, "ATR_DELTA"));
                    row.setErDelta(getSafeDouble(tokens, colMap, "ER_DELTA"));
                    row.setYieldSpreadDelta(getSafeDouble(tokens, colMap, "YIELD_SPREAD_DELTA"));

                    // Context
                    row.setEurusdAtr(getSafeDouble(tokens, colMap, "EURUSD_ATR"));
                    row.setAudjpyAtr(getSafeDouble(tokens, colMap, "AUDJPY_ATR"));
                    row.setEurjpyAtr(getSafeDouble(tokens, colMap, "EURJPY_ATR"));

                    // Note: Shocks/Strategy variables will default to 0.0 here and get updated inside ModelTrainer loop.

                    dataset.add(row);
                } catch (Exception e) {
                    System.err.println("Warning: Skipped entry line " + lineNumber + " due to parsing breakdown: " + e.getMessage());
                }
            }
        }

        return dataset;
    }

    private static double getSafeDouble(String[] tokens, Map<String, Integer> colMap, String columnName) {
        if (!colMap.containsKey(columnName)) {
            return 0.0;
        }
        int index = colMap.get(columnName);
        if (index >= tokens.length || tokens[index].trim().isEmpty()) {
            return 0.0;
        }
        // Handles scientific notation automatically (e.g., 7.367085E-5)
        return Double.parseDouble(tokens[index].trim());
    }

    public static void main(String[] args) {
        try {
            String path = "data/assembled_dataset.csv";
            List<MarketData> dataList = loadDataset(path);
            System.out.println("Loader Success! Total observations initialized: " + dataList.size());

            if (!dataList.isEmpty()) {
                MarketData testRow = dataList.get(0);
                System.out.println("\nVerification Scan on Row 1:");
                System.out.println("Timestamp: " + testRow.getTimestamp());
                System.out.println("USD/JPY Volatility (ATR): " + testRow.getUsdjpyAtr());
                System.out.println("US Bond Coordinate: " + testRow.getUsBondLevel());
                System.out.println("ATR Momentum Trajectory: " + testRow.getAtrDelta());
            }
        } catch (IOException e) {
            System.err.println("Failed to read the dataset file: " + e.getMessage());
        }
    }


}