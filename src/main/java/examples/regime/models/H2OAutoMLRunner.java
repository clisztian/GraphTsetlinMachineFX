package examples.regime.models;

import ai.h2o.automl.AutoML;
import ai.h2o.automl.AutoMLBuildSpec;
import water.H2O;
import water.Key;
import water.fvec.Frame;
import water.fvec.NFSFileVec;
import water.parser.ParseDataset;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class H2OAutoMLRunner {

    public static void main(String[] args) throws IOException {
        String sourceCsv = "labeled_dataset.csv";
        String trainCsv = "temp_train_regimes.csv";
        String testCsv = "temp_test_regimes.csv";

        Thread.currentThread().setContextClassLoader(H2OAutoMLRunner.class.getClassLoader());


            // 1. Execute strict sequential chronological data splitting (80% Train / 20% Test)
            splitDatasetSequentially(sourceCsv, trainCsv, testCsv, 0.80);


// 2. EMBEDDED STARTUP FIX: Configure H2O directly within the existing JVM thread
            System.out.println("Initializing embedded H2O Local Cluster Engine...");

// Define cluster arguments to run completely local without seeking external network nodes
            String[] h2oArgs = new String[] {
                    "-ip", "127.0.0.1",
                    "-port", "54321",
                    "-disable_web",     // Speeds up booting by skipping the UI interface
                    "-quiet"            // Tones down internal verbose console prints
            };

// Start core engine safely without letting it take over or kill the JVM process
            water.H2OApp.main(new String[]{}); // disable web UI for simplicity
            H2O.waitForCloudSize(1, 10_000);

            System.out.println("Waiting for local cloud node registry allocation...");
            while (H2O.getCloudSize() < 1 || H2O.SELF == null) {
                try {
                    Thread.sleep(500);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException("H2O local cluster initialization aborted.", ie);
                }
            }
            System.out.println("Cluster successfully stabilized. Cloud Node: " + H2O.SELF.toString());

// 3. Parse data frames safely from generated local files
            System.out.println("Parsing training data frame into memory store...");

// FIX: Pause the application main execution thread to wait for cloud stabilization
            System.out.println("Waiting for local cloud node registry allocation...");
            while (H2O.getCloudSize() < 1 || H2O.SELF == null) {
                try {
                    Thread.sleep(500); // Check node health status every 500ms
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException("H2O local cluster initialization aborted.", ie);
                }
            }
            System.out.println("Cluster successfully stabilized. Cloud Node: " + H2O.SELF.toString());

// 3. Parse data frames from generated local files safely
            System.out.println("Parsing training data frame into memory store...");
            Frame trainFrame = ParseDataset.parse(
                    Key.make("train_regimes.hex"),
                    new Key[]{NFSFileVec.make(new File(trainCsv))._key}
            );

            System.out.println("Parsing out-of-sample evaluation frame...");
            Frame testFrame = ParseDataset.parse(
                    Key.make("test_regimes.hex"),
                    new Key[]{NFSFileVec.make(new File(testCsv))._key}
            );

            // 4. Force target column metadata from numerical to Categorical Factor enum
            String responseColumn = "LABEL";
            trainFrame.replace(trainFrame.find(responseColumn), trainFrame.vec(responseColumn).toCategoricalVec());
            testFrame.replace(testFrame.find(responseColumn), testFrame.vec(responseColumn).toCategoricalVec());

            water.DKV.put(trainFrame);
            water.DKV.put(testFrame);

            // 5. Construct AutoML Optimization Grid Criteria Specifications
            AutoMLBuildSpec buildSpec = new AutoMLBuildSpec();
            buildSpec.input_spec.training_frame = trainFrame._key;
            buildSpec.input_spec.response_column = responseColumn;

            // Drop timestamp string attributes to clean up calculation rows
            buildSpec.input_spec.ignored_columns = new String[]{"Timestamp"};

            // Bound processing resource parameters
            buildSpec.build_control.stopping_criteria.set_max_runtime_secs(180); // 3 minute optimization matrix limit
            buildSpec.build_control.nfolds = 5; // 5-Fold cross validation on train split

            // 6. Orchestrate optimization run sweeps
            System.out.println("Beginning H2O AutoML optimization sweeps...");
            AutoML aml = AutoML.startAutoML(buildSpec);
            aml.get(); // Block thread until leaderboard converges

        // =====================================================================
// 7. ITERATE AND SCORE ALL MODELS IN THE LEADERBOARD
// =====================================================================
        System.out.println("\n=== Iterating and Scoring All AutoML Models Out-Of-Sample ===");
        // Bypasses the Leaderboard type by converting its key to a string look-up
        Key[] modelKeys = aml.leaderboard().getModelKeys();

// Open a file writer to save all competitive metrics directly to a CSV
        try (BufferedWriter csvWriter = new BufferedWriter(new FileWriter("automl_all_models_oos.csv"))) {
            csvWriter.write("Model_ID,Class,Precision,Recall,F1_Score,Overall_Accuracy\n");

            // Loop directly over the keys array
            for (int m = 0; m < modelKeys.length; m++) {
                Key modelKey = modelKeys[m];
                String modelId = modelKey.toString();

                System.out.println("Deploying OOS Eval for Model [" + (m+1) + "/" + modelKeys.length + "]: " + modelId);

                // Fetch the corresponding model from H2O's key store safely
                hex.Model<?,?,?> model = water.DKV.getGet(modelKey);
                if (model == null) continue;

                // Score the model against the raw, untouched out-of-sample test split
                Frame predictions = model.score(testFrame);

                // Calculate the individual class breakdowns
                evaluateAndLogMetrics(modelId, testFrame, predictions, csvWriter);

                predictions.delete(); // Clear temporary frame memory
            }
        }
        System.out.println("All model out-of-sample evaluation metrics exported to 'automl_all_models_oos.csv'!");




            // 7. Render Model Ranking Leaderboard using native string tables
            System.out.println("\n=== H2O AutoML Model Leaderboard ===");
            System.out.println(aml.leaderboard().toString());

            // 8. Deploy Winner algorithm onto unseen Out-Of-Sample Target Test sets
            System.out.println("\nEvaluating optimal leader model performance metrics...");
            Frame predictions = aml.leader().score(testFrame);
            evaluateConfusionMatrix(testFrame, predictions);


    }





    private static void splitDatasetSequentially(String sourcePath, String trainPath, String testPath, double trainRatio) throws IOException {
        System.out.println("Splitting " + sourcePath + " chronologically with 10% Class 0 training downsampling...");
        List<String> rows = new ArrayList<>();
        String header = "";

        // 1. Read all rows into memory
        try (BufferedReader br = new BufferedReader(new FileReader(sourcePath))) {
            header = br.readLine();
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    rows.add(line);
                }
            }
        }

        int trainSize = (int) (rows.size() * trainRatio);
        java.util.Random rand = new java.util.Random(42); // Seeded for scientific reproducibility

        // 2. Export Training Partition with Class 0 Downsampling
        int originalTrainCount = 0;
        int retainedTrainCount = 0;

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(trainPath))) {
            bw.write(header);
            bw.newLine();

            for (int i = 0; i < trainSize; i++) {
                String row = rows.get(i);
                originalTrainCount++;

                // Extract the label column (assuming it's the final token in the comma-separated string)
                String[] tokens = row.split(",");
                int label = Integer.parseInt(tokens[tokens.length - 1].trim());

                if (label == 0) {
                    // Skips 90% of Class 0 rows randomly, keeping only a 10% sample
                    if (rand.nextDouble() < 0.10) {
                        bw.write(row);
                        bw.newLine();
                        retainedTrainCount++;
                    }
                } else {
                    // Keep 100% of volatile regimes (Classes 1, 2, 3)
                    bw.write(row);
                    bw.newLine();
                    retainedTrainCount++;
                }
            }
        }
        System.out.printf("Training optimization subset balanced: Reduced from %d to %d rows.%n", originalTrainCount, retainedTrainCount);

        // 3. Export Test Partition (KEEP RAW / UNTOUCHED)
        // CRITICAL: Never downsample your out-of-sample testing validation frame.
        // It must remain identical to real-world market distributions to keep your paper metrics accurate.
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(testPath))) {
            bw.write(header);
            bw.newLine();
            for (int i = trainSize; i < rows.size(); i++) {
                bw.write(rows.get(i));
                bw.newLine();
            }
        }
    }


    private static void evaluateAndLogMetrics(String modelId, Frame testFrame, Frame predictions, BufferedWriter csvWriter) throws IOException {
        long totalRows = testFrame.numRows();
        int labelIdx = testFrame.find("LABEL");
        int predIdx = predictions.find("predict");

        int[][] cm = new int[4][4];
        int totalCorrect = 0;

        for (long row = 0; row < totalRows; row++) {
            int actual = (int) testFrame.vec(labelIdx).at8(row);
            int predicted = (int) predictions.vec(predIdx).at8(row);
            if (actual >= 0 && actual < 4 && predicted >= 0 && predicted < 4) {
                cm[actual][predicted]++;
                if (actual == predicted) totalCorrect++;
            }
        }

        double overallAcc = (double) totalCorrect / totalRows;

        // Compute Precision, Recall, and F1 for each of the 4 regimes
        for (int i = 0; i < 4; i++) {
            int actualTotal = 0;   // Row sum (for Recall)
            int predictedTotal = 0; // Column sum (for Precision)

            for (int j = 0; j < 4; j++) {
                actualTotal += cm[i][j];
                predictedTotal += cm[j][i];
            }

            double recall = (actualTotal > 0) ? (double) cm[i][i] / actualTotal : 0.0;
            double precision = (predictedTotal > 0) ? (double) cm[i][i] / predictedTotal : 0.0;
            double f1 = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0.0;

            // Write row out to the structured dataset report
            csvWriter.write(String.format("%s,%d,%.5f,%.5f,%.5f,%.5f\n", modelId, i, precision, recall, f1, overallAcc));
        }
    }


//
//    private static void splitDatasetSequentially(String sourcePath, String trainPath, String testPath, double trainRatio) throws IOException {
//        System.out.println("Splitting " + sourcePath + " chronologically (" + (trainRatio*100) + "/" + ((1-trainRatio)*100) + ")...");
//        List<String> rows = new ArrayList<>();
//        String header = "";
//
//        try (BufferedReader br = new BufferedReader(new FileReader(sourcePath))) {
//            header = br.readLine();
//            String line;
//            while ((line = br.readLine()) != null) {
//                if (!line.trim().isEmpty()) {
//                    rows.add(line);
//                }
//            }
//        }
//
//        int trainSize = (int) (rows.size() * trainRatio);
//
//        // Export training partition file blocks
//        try (BufferedWriter bw = new BufferedWriter(new FileWriter(trainPath))) {
//            bw.write(header);
//            bw.newLine();
//            for (int i = 0; i < trainSize; i++) {
//                bw.write(rows.get(i));
//                bw.newLine();
//            }
//        }
//
//        // Export validation partition testing file blocks
//        try (BufferedWriter bw = new BufferedWriter(new FileWriter(testPath))) {
//            bw.write(header);
//            bw.newLine();
//            for (int i = trainSize; i < rows.size(); i++) {
//                bw.write(rows.get(i));
//                bw.newLine();
//            }
//        }
//    }

    private static void evaluateConfusionMatrix(Frame testFrame, Frame predictions) {
        long totalRows = testFrame.numRows();
        int labelIdx = testFrame.find("LABEL");
        int predIdx = predictions.find("predict");

        int[][] confusionMatrix = new int[4][4];
        int totalCorrect = 0;

        for (long row = 0; row < totalRows; row++) {
            int actual = (int) testFrame.vec(labelIdx).at8(row);
            int predicted = (int) predictions.vec(predIdx).at8(row);

            if (actual >= 0 && actual < 4 && predicted >= 0 && predicted < 4) {
                confusionMatrix[actual][predicted]++;
                if (actual == predicted) {
                    totalCorrect++;
                }
            }
        }

        System.out.printf("Overall H2O AutoML OOS Test Accuracy: %.4f%n", (double) totalCorrect / totalRows);
        System.out.println("----------------------------------------------");
        for (int i = 0; i < 4; i++) {
            int rowTotal = 0;
            for (int j = 0; j < 4; j++) rowTotal += confusionMatrix[i][j];

            double classAcc = (rowTotal > 0) ? (double) confusionMatrix[i][i] / rowTotal : 0.0;
            System.out.printf("Class %d Accuracy: %.4f (Base Group Size: %d)%n", i, classAcc, rowTotal);
        }
    }
}
