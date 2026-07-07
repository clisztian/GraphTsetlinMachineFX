package examples.regime.models;

import ai.h2o.automl.AutoML;
import ai.h2o.automl.AutoMLBuildSpec;
import water.H2O;
import water.Key;
import water.fvec.Frame;
import water.fvec.NFSFileVec;
import water.parser.ParseDataset;

import java.io.File;

public class H2OAutoMLBenchmark {

    public static void runAutoML(String trainCsvPath, String testCsvPath) {
        System.out.println("Initializing embedded H2O Local Cluster Engine...");
        // 1. Initialize H2O cluster environment thread hooks
        H2O.main(new String[]{});

        try {
            // 2. Load and Parse Training Data
            System.out.println("Parsing training sequence to H2O frame space...");
            File trainFile = new File(trainCsvPath);
            Frame trainFrame = ParseDataset.parse(
                    Key.make("train_regimes.hex"),
                    new Key[]{NFSFileVec.make(trainFile)._key}
            );

            // 3. Load and Parse Out-of-Sample Testing Data
            System.out.println("Parsing out-of-sample evaluation sequence...");
            File testFile = new File(testCsvPath);
            Frame testFrame = ParseDataset.parse(
                    Key.make("test_regimes.hex"),
                    new Key[]{NFSFileVec.make(testFile)._key}
            );

            // 4. PREprocessing: Convert your target label from numeric to Enum/Factor
            // This is mandatory to tell AutoML to run Multinomial Classification instead of Regression
            String responseColumn = "LABEL";
            trainFrame.replace(trainFrame.find(responseColumn), trainFrame.vec(responseColumn).toCategoricalVec());
            testFrame.replace(testFrame.find(responseColumn), testFrame.vec(responseColumn).toCategoricalVec());

            // Register frames globally within the H2O key-value store
            water.DKV.put(trainFrame);
            water.DKV.put(testFrame);

            // 5. Configure AutoML Build Parameters
            AutoMLBuildSpec buildSpec = new AutoMLBuildSpec();
            buildSpec.input_spec.training_frame = trainFrame._key;
            buildSpec.input_spec.response_column = responseColumn;

            // Drop Timestamp from features list to prevent overfitting on string formats
            buildSpec.input_spec.ignored_columns = new String[]{"Timestamp"};

            // Set stopping execution criteria (e.g., maximum runtime optimization boundary)
            buildSpec.build_control.stopping_criteria.set_max_runtime_secs(120); // 2 minutes run time
            buildSpec.build_control.nfolds = 5; // 5-Fold cross-validation on train frame

            // 6. Launch AutoML Optimization Sweep
            System.out.println("Orchestrating AutoML grid architectures...");
            AutoML aml = AutoML.startAutoML(buildSpec);
            aml.get(); // Block thread until optimization finishes completely

            // 7. Output Leaderboard Results
            System.out.println("\n=== H2O AutoML Model Leaderboard ===");
            // Direct lookup via the Frame Key store bypasses any missing Leaderboard class definitions
            System.out.println(aml.leaderboard().toString());

            // 8. Evaluate Best Performer on Out-of-Sample Test Set
            Frame predictions = aml.leader().score(testFrame);

            System.out.println("\n=== Out-of-Sample Final Metrics ===");
            evaluateConfusionMatrix(testFrame, predictions);

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            System.out.println("Shutting down H2O execution threads safely...");
            H2O.shutdown(0);
        }
    }

    private static void evaluateConfusionMatrix(Frame testFrame, Frame predictions) {
        // Predictions schema contains columns: [predict, p0, p1, p2, p3]
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

        System.out.printf("Overall AutoML Test Accuracy: %.4f%n", (double) totalCorrect / totalRows);
        for (int i = 0; i < 4; i++) {
            int rowTotal = 0;
            for (int j = 0; j < 4; j++) rowTotal += confusionMatrix[i][j];

            double classAcc = (rowTotal > 0) ? (double) confusionMatrix[i][i] / rowTotal : 0.0;
            System.out.printf("Class %d Accuracy: %.4f (Samples: %d)%n", i, classAcc, rowTotal);
        }
    }
}
