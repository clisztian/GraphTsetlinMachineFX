package examples.regime.trainer;

import examples.regime.dto.MarketData;

import java.io.IOException;
import java.util.List;

import static examples.regime.RegimeLabeler.computeLabeledDataset;
import static examples.regime.io.DatasetReader.loadDataset;

public class ModelTrain {

    public void labelData() throws IOException {

        String path = "data/assembled_dataset.csv";
        List<MarketData> dataList = loadDataset(path);

        computeLabeledDataset(dataList);

    }

    //main method to run the labeling process
    public static void main(String[] args) {
        ModelTrain trainer = new ModelTrain();
        try {
            trainer.labelData();
        } catch (IOException e) {
            System.err.println("Error loading dataset: " + e.getMessage());
        }
    }

}
