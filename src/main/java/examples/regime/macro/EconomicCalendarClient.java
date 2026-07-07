package examples.regime.macro;


import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

public class EconomicCalendarClient {

    private static final String API_URL = "https://financialmodelingprep.com/api/v3/economic_calendar";
    private final String apiKey;
    private final HttpClient httpClient;

    // FMP formats dates as 'yyyy-MM-dd HH:mm:ss' or standard ISO strings
    private static final DateTimeFormatter EVENT_TIME_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    public EconomicCalendarClient(String apiKey) {
        this.apiKey = apiKey;
        this.httpClient = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();
    }

    /**
     * Executes chronological sliding queries to scrape deep macro statistics.
     * @param startYear The earliest boundary (e.g., 2016)
     * @param endYear The current boundary (e.g., 2026)
     */
    public List<MacroShockProcessor.EconomicReleaseEvent> fetchHistoricalInflationShocks(int startYear, int endYear) {
        List<MacroShockProcessor.EconomicReleaseEvent> macroEvents = new ArrayList<>();
        LocalDate cursorDate = LocalDate.of(startYear, 1, 1);
        LocalDate targetEnd = LocalDate.of(endYear, 12, 31);

        System.out.println("Beginning 10-year macroeconomic data ingestion sequence...");

        // FMP requests accept max 3-month blocks to prevent buffer overflows
        while (cursorDate.isBefore(targetEnd)) {
            LocalDate chunkEnd = cursorDate.plusMonths(3);
            if (chunkEnd.isAfter(targetEnd)) {
                chunkEnd = targetEnd;
            }

            try {
                List<MacroShockProcessor.EconomicReleaseEvent> chunkData = fetchCalendarChunk(cursorDate, chunkEnd);
                macroEvents.addAll(chunkData);

                // Be polite to API rate limits
                Thread.sleep(150);
            } catch (Exception e) {
                System.err.println("Warning: Data chunk failed between " + cursorDate + " and " + chunkEnd + " -> " + e.getMessage());
            }

            cursorDate = chunkEnd.plusDays(1);
        }

        System.out.println("Ingestion Complete. Extracted " + macroEvents.size() + " total relevant inflation data releases.");
        return macroEvents;
    }

    /**
     * Pulls and parses a single isolated 90-day structural calendar window.
     */
    private List<MacroShockProcessor.EconomicReleaseEvent> fetchCalendarChunk(LocalDate from, LocalDate to) throws Exception {
        List<MacroShockProcessor.EconomicReleaseEvent> chunkList = new ArrayList<>();

        String urlString = String.format("%s?from=%s&to=%s&apikey=%s",
                API_URL, from.toString(), to.toString(), this.apiKey);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(urlString))
                .GET()
                .header("Accept", "application/json")
                .build();

        HttpResponse<String> response = this.httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new RuntimeException("HTTP Server Error code: " + response.statusCode());
        }

        String body = response.body().trim();
        if (body.length() <= 2) {
            return chunkList; // Empty array payload "[]"
        }

        // Split the bulk array into individual JSON object segments safely
        String[] jsonObjects = body.split("\\},\\{");

        for (String objStr : jsonObjects) {
            // Normalize edge boundaries from array splitting anomalies
            String normalizedObj = objStr.replace("[{", "").replace("}]", "");

            String country = extractJsonStringField(normalizedObj, "country");
            String eventName = extractJsonStringField(normalizedObj, "event").toUpperCase();

            // 1. Structural Filter - Isolate target macro shocks
            if (("US".equals(country) || "JP".equals(country)) &&
                    (eventName.contains("CPI") || eventName.contains("INFLATION") || eventName.contains("CORE CPI"))) {

                // 2. Extract continuous value pairs safely
                String actualStr = extractJsonRawField(normalizedObj, "actual");
                String estimateStr = extractJsonRawField(normalizedObj, "estimate");

                // Skip unannounced consensus variables or blank target gaps
                if (actualStr == null || estimateStr == null || "null".equals(actualStr) || "null".equals(estimateStr)) {
                    continue;
                }

                double actual = Double.parseDouble(actualStr);
                double estimate = Double.parseDouble(estimateStr);

                // Extract and clean timestamp indices
                String dateStr = extractJsonStringField(normalizedObj, "date");
                if (dateStr.length() >= 19) {
                    dateStr = dateStr.replace("T", " ").substring(0, 19);
                    LocalDateTime releaseTime = LocalDateTime.parse(dateStr, EVENT_TIME_FORMATTER);
                    String currency = "US".equals(country) ? "USD" : "JPY";

                    chunkList.add(new MacroShockProcessor.EconomicReleaseEvent(releaseTime, eventName, currency, actual, estimate));
                }
            }
        }

        return chunkList;
    }

    /**
     * Native helper to extract string literals bounded by quotes from a raw JSON row snippet.
     */
    private static String extractJsonStringField(String json, String key) {
        String searchKey = "\"" + key + "\":\"";
        int startIdx = json.indexOf(searchKey);
        if (startIdx == -1) {
            // Fallback for case where numeric variables are mistakenly quoted
            return extractJsonRawField(json, key);
        }
        startIdx += searchKey.length();
        int endIdx = json.indexOf("\"", startIdx);
        return (endIdx != -1) ? json.substring(startIdx, endIdx).trim() : "";
    }

    /**
     * Native helper to extract raw numeric/null literal features from a JSON row snippet.
     */
    private static String extractJsonRawField(String json, String key) {
        String searchKey = "\"" + key + "\":";
        int startIdx = json.indexOf(searchKey);
        if (startIdx == -1) return null;

        startIdx += searchKey.length();
        int endIdx = json.length();

        // Find nearest structural boundary character separating fields
        for (int i = startIdx; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == ',' || c == '}' || c == '"') {
                endIdx = i;
                break;
            }
        }
        return json.substring(startIdx, endIdx).trim();
    }

    public static void main(String[] args) {
        // Quick integration check
        String myDemoKey = "";
        EconomicCalendarClient client = new EconomicCalendarClient(myDemoKey);

        // Let's test sweeping a past chunk interval
        List<MacroShockProcessor.EconomicReleaseEvent> results = client.fetchHistoricalInflationShocks(2017, 2018);
        if(!results.isEmpty()) {
            System.out.println("Sample entry captured: " + results.get(0).eventName + " on " + results.get(0).releaseTime);
        }
    }
}
