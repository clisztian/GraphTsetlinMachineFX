package examples.regime;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class ModelScatterEvaluationMatrix extends Application {

    private static final String CSV_FILE = "automl_all_models_oos.csv";

    @Override
    public void start(Stage stage) {
        stage.setTitle("Out-of-Sample Strategic Risk Evaluation Matrix");

        // 1. Establish Structured Production Axes (Locked from 0% to 100%)
        NumberAxis xAxis = new NumberAxis(0, 100, 10);
        xAxis.setLabel("Class 1 (Trending Low Vol) Recall %");

        NumberAxis yAxis = new NumberAxis(0, 100, 10);
        yAxis.setLabel("Class 2 (Choppy High Vol) Recall %");

        // 2. Instantiate Scatter Chart Component
        ScatterChart<Number, Number> scatterChart = new ScatterChart<>(xAxis, yAxis);
        scatterChart.setTitle("Strategic Regime Core Tracking Matrix: GraphTM vs. H2O AutoML");
        scatterChart.setAnimated(false);

        // 3. Define the Competitive Baselines Series Groups
        XYChart.Series<Number, Number> competitorSeries = new XYChart.Series<>();
        competitorSeries.setName("H2O AutoML Ensembles Frameworks");

        XYChart.Series<Number, Number> graphTmSeries = new XYChart.Series<>();
        graphTmSeries.setName("GraphTM (Our Proposed Architecture)");

        // 4. Inject GraphTM Coordinates manually
        XYChart.Data<Number, Number> graphTmData = new XYChart.Data<>(24.16107, 71.91919);
        graphTmSeries.getData().add(graphTmData);

        // 5. Parse out-of-sample CSV data arrays
        Map<String, double[]> autoMlModels = parseAutoMlCsv(CSV_FILE);
        for (Map.Entry<String, double[]> entry : autoMlModels.entrySet()) {
            String labelName = cleanModelId(entry.getKey());
            double[] recalls = entry.getValue(); // index 0 = Class 1, index 1 = Class 2

            XYChart.Data<Number, Number> dataPoint = new XYChart.Data<>(recalls[0] * 100, recalls[1] * 100);
            competitorSeries.getData().add(dataPoint);
        }

        // Add both series contexts to chart architecture
        scatterChart.getData().addAll(competitorSeries, graphTmSeries);

        // 6. Apply UI tooltips to nodes to allow data point inspection on hover
        stage.setOnShown(ev -> {
            for (XYChart.Data<Number, Number> data : competitorSeries.getData()) {
                Tooltip.install(data.getNode(), new Tooltip("AutoML Ensemble\nTrend Rec: " + String.format("%.2f%%", data.getXValue()) + "\nChoppy Rec: " + String.format("%.2f%%", data.getYValue())));
            }
            Tooltip.install(graphTmData.getNode(), new Tooltip("GraphTM Full Model\nTrend Rec: 24.16%\nChoppy Rec: 71.92%"));
        });

        // 7. Render Layout Frame Component with High-Contrast Theme Layer overrides
        VBox vbox = new VBox(scatterChart);
        VBox.setVgrow(scatterChart, javafx.scene.layout.Priority.ALWAYS);
        Scene scene = new Scene(vbox, 1000, 750);

        scatterChart.setStyle(
                "-fx-font-size: 13px;" +
                        "-fx-font-family: 'Helvetica', 'Arial', sans-serif;"
        );

// Style the chart's structural sub-components using standard CSS string sheets
        String cssStyles =
                ".chart-title { -fx-font-size: 16px; -fx-font-weight: bold; -fx-text-fill: #111111; }" +
                        ".axis-label { -fx-font-size: 12px; -fx-font-weight: bold; }" +
                        // AutoML models: Muted slate circles
                        ".series0.chart-bubble { -fx-background-color: #7f8c8d; -fx-background-radius: 5px; -fx-padding: 5px; }" +
                        // GraphTM: Prominent red-gold diamond beacon
                        ".series1.chart-bubble { -fx-background-color: #d35400, #f1c40f; -fx-background-insets: 0, 2; -fx-shape: 'M 0 -5 L 5 0 L 0 5 L -5 0 Z'; -fx-padding: 8px; }" +
                        // Strategy Risk Matrix Background Gradient (Safe zone is top-right)
                        ".chart-plot-background { -fx-background-color: linear-gradient(to top right, #fce4ec, #e8f5e9); }";

// Append the styles to the scene's stylesheet manager using an inline string representation safely
        vbox.setStyle(cssStyles);
        scatterChart.getStylesheets().add("data:text/css," + cssStyles.replace("%", "%25"));

        stage.setScene(scene);
        stage.show();
    }

    private Map<String, double[]> parseAutoMlCsv(String filePath) {
        Map<String, double[]> modelMap = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line = br.readLine(); // skip header
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] tokens = line.split(",");
                String modelId = tokens[0].trim();
                int targetClass = Integer.parseInt(tokens[1].trim());
                double recall = Double.parseDouble(tokens[3].trim());

                if (targetClass == 1 || targetClass == 2) {
                    modelMap.putIfAbsent(modelId, new double[2]);
                    if (targetClass == 1) modelMap.get(modelId)[0] = recall;
                    else modelMap.get(modelId)[1] = recall;
                }
            }
        } catch (IOException | NumberFormatException e) {
            System.err.println("CSV Parsing Error: " + e.getMessage());
        }
        return modelMap;
    }

    private String cleanModelId(String modelId) {
        if (modelId.contains("_AutoML_")) return modelId.substring(0, modelId.indexOf("_AutoML_"));
        return modelId;
    }

    public static void main(String[] args) {
        launch(args);
    }
}
