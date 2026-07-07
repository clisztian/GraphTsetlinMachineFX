package examples.regime.trainer;

import examples.regime.dto.MarketData;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TrainingWeightCalibrator {

    public static Map<Integer, Double> calculateClassUpdateProbabilities(List<MarketData> trainSet) {
        int[] counts = new int[4];
        int validTotal = 0;

        for (MarketData data : trainSet) {
            int label = data.getRegimeLabel();
            if (label >= 0 && label <= 3) {
                counts[label]++;
                validTotal++;
            }
        }

        // Find the count of the rarest class to use as our base anchor
        int rarestCount = Integer.MAX_VALUE;
        for (int count : counts) {
            if (count > 0 && count < rarestCount) {
                rarestCount = count;
            }
        }

        Map<Integer, Double> updateProbabilities = new HashMap<>();
        System.out.println("--- Training Class Balancing Map ---");
        for (int i = 0; i < 4; i++) {
            // Probability = Rarest Count / This Class Count
            // This guarantees the rarest class always has a 1.0 (100%) update probability
            double pUpdate = (double) rarestCount / counts[i];
            updateProbabilities.put(i, pUpdate);

            System.out.printf("Regime %d: Size = %5d | Automata Update Probability: %.4f\n",
                    i, counts[i], pUpdate);
        }

        return updateProbabilities;
    }
}