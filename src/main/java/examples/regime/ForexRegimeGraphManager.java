package examples.regime;

import encoding.embedding.FixedLevelEmbedding;
import encoding.embedding.LinearEmbedding;
import encoding.hypervector.SparseBinaryVector;
import examples.regime.dto.MarketData;
import examples.regime.io.DatasetReader;
import graph.TsetlinAttentionGraph;
import graph.TsetlinEdge;
import graph.TsetlinVertex;
import util.MutableInt;

import java.io.IOException;
import java.util.*;

import static examples.regime.RegimeLabeler.computeLabeledDataset;

/**
 * A manager class for constructing and training a Tsetlin Attention Graph for Forex regime detection.
 *
 * Builds the graph structure, initializes nodes and edges with appropriate embeddings,
 * and provides methods for adding samples and learning from the data.
 *
 */

public class ForexRegimeGraphManager {

    private final TsetlinAttentionGraph graph;

    Random random = new Random();

    public TsetlinAttentionGraph getGraph() {
        return graph;
    }

    // Define the embeddings for the various market features
    private final LinearEmbedding priceEmbed = new LinearEmbedding(-0.50, 0.50, 50);
    private final LinearEmbedding atrEmbed   = new LinearEmbedding(.0001, 0.008, 20); // % ATR
    private final LinearEmbedding erEmbed    = new LinearEmbedding(0.0, 1.0, 10);    // ER [0,1]
    private final LinearEmbedding bondEmbed  = new LinearEmbedding(50.0, 120.0, 40);
    private final LinearEmbedding jpbondEmbed  = new LinearEmbedding(40.0, 70.0, 40);

    // Delta Embeddings (Centered at 0.0 to capture +/- trajectory)
    private final LinearEmbedding deltaAtrEmbed = new LinearEmbedding(-0.0020, 0.0020, 20);
    private final LinearEmbedding deltaErEmbed  = new LinearEmbedding(-1.0, 1.0, 20);
    private final LinearEmbedding deltaYieldEmbed = new LinearEmbedding(-4.0, 5.0, 20);

    //WTI Oil Price Embedding
    private final LinearEmbedding oilEmbed = new LinearEmbedding(10.0, 130.0, 30);

    private final FixedLevelEmbedding newsShockEmbed = new FixedLevelEmbedding(4); // 5 levels of news shock severity

    private final Map<String, TsetlinVertex> nodes = new HashMap<>();

    /**
     * Constructor for ForexRegimeGraphManager.
     *
     * Initializes the Tsetlin Attention Graph, adds nodes representing various market features,
     * and establishes edges between nodes to capture causal relationships and dependencies.
     * Different weights and types of edges are used to represent the influence of one feature on another.
     */
    public ForexRegimeGraphManager() {
        this.graph = new TsetlinAttentionGraph(2);


        addNode("USDJPY_PRICE", 1.1f);
        addNode("USDJPY_ATR", 1.2f); // High importance: volatility defines the regime


        addNode("US_BOND_LEVEL", 0.83f);
        addNode("JP_BOND_LEVEL", 0.84f);
        addNode("WTI_OIL", 0.75f);


        addNode("ATR_DELTA", 0.96f);
        addNode("ER_DELTA", 0.97f);
        addNode("YIELD_SPREAD_DELTA", 0.98f);
        //addNode("STRATEGY_PERF", 1.05f);


        addNode("EURUSD_ATR", 0.71f);
        addNode("AUDJPY_VOL", 0.82f);

        addNode("USD_NEWS_SHOCK", 0.91f);

        link("US_BOND_LEVEL", "USDJPY_PRICE", "Level_Causation", 0.81f);
        link("JP_BOND_LEVEL", "USDJPY_PRICE", "Level_Causation", 0.82f);
        link("YIELD_SPREAD_DELTA", "USDJPY_PRICE", "Momentum_Causation", 0.953f);
        link("WTI_OIL", "USDJPY_PRICE", "Trade_Causation", 0.74f);
        link("ER_DELTA", "USDJPY_PRICE", "Signal_Validation", 0.855f);
        link("ATR_DELTA", "USDJPY_PRICE", "Volatility_Validation", 0.756f);
        link("USDJPY_ATR", "USDJPY_PRICE", "Risk_Scaling", 0.77f);


        link("USD_NEWS_SHOCK", "USDJPY_PRICE", "Inflationary_Impulse", 0.88f);


        link("YIELD_SPREAD_DELTA", "USDJPY_ATR", "Policy_Uncertainty", 0.87f);
        link("ATR_DELTA", "USDJPY_ATR", "Vol_Uncertainty", 0.97f);
        link("AUDJPY_VOL", "USDJPY_ATR", "Risk_Sentiment_Transmission", 0.856f);
        link("EURUSD_ATR", "USDJPY_ATR", "Arb_Volatility_Uncertainty", 0.757f);
        link("USDJPY_PRICE", "USDJPY_ATR", "Reflexive_Feedback", 0.64f);


        link("USD_NEWS_SHOCK", "USDJPY_ATR", "Exogenous_Regime_Trigger", 0.96f);


        link("YIELD_SPREAD_DELTA", "ER_DELTA", "Fundamental_Clarity", 0.623f);


        link("USD_NEWS_SHOCK", "YIELD_SPREAD_DELTA", "Curve_Repricing_Trigger", 0.843f);



    }

    private void addNode(String name, float importance) {
        TsetlinVertex vertex = new TsetlinVertex(name, importance);
        nodes.put(name, vertex);
        graph.insertVertex(vertex);
    }

    private void link(String from, String to, String type, float weight) {
        // Edge hypervectors act as the relational variable
        TsetlinEdge edge = new TsetlinEdge(type, weight, SparseBinaryVector.randVector());
        graph.insertEdge(nodes.get(from), nodes.get(to), edge);
    }

    /**
     * Adds a list of MarketData samples to the learning dataset, initializing the embeddings for each feature.
     * @param trainingData
     * @return
     */
    public List<MarketData> addSample(List<MarketData> trainingData) {

        List<MarketData> learningList = new ArrayList<>();

        Map<Integer, MutableInt> classCounts = new HashMap<>();

        //use the previous label (prediction from 72 hours prior) to build a baseline prediction distribution for each class
        Map<Integer, MutableInt> baselinePredictions = new HashMap<>();

        for(int i = 0; i < trainingData.size(); i++) {

            MarketData data = trainingData.get(i);

            int regimeLabel = data.getRegimeLabel();

            //if regimeLabel not 0,1,2,3 skip
            if(regimeLabel < 0 || regimeLabel > 3) {
                System.out.println("Skipping sample at index " + i + " with invalid regime label " + regimeLabel);
                continue;
            }

            learningList.add(data);



            // --- Initialize Layer 0 Encodings ---
            nodes.get("USDJPY_PRICE").initializeLayerData(priceEmbed.forward(data.getUsdjpyPrice()));
            nodes.get("USDJPY_ATR").initializeLayerData(atrEmbed.forward(data.getUsdjpyAtr()));

            nodes.get("US_BOND_LEVEL").initializeLayerData(bondEmbed.forward(data.getUsBondLevel()));
            nodes.get("JP_BOND_LEVEL").initializeLayerData(jpbondEmbed.forward(data.getJpBondLevel()));
            nodes.get("WTI_OIL").initializeLayerData(oilEmbed.forward(data.getWtiOil()));

            // Delta Encodings (Momentum)
            nodes.get("ATR_DELTA").initializeLayerData(deltaAtrEmbed.forward(data.getAtrDelta()));
            nodes.get("ER_DELTA").initializeLayerData(deltaErEmbed.forward(data.getErDelta()));
            nodes.get("YIELD_SPREAD_DELTA").initializeLayerData(deltaYieldEmbed.forward(data.getYieldSpreadDelta()));

            // Cross-Pair ATRs
            nodes.get("EURUSD_ATR").initializeLayerData(atrEmbed.forward(data.getEurusdAtr()));
            nodes.get("AUDJPY_VOL").initializeLayerData(atrEmbed.forward(data.getAudjpyAtr()));

            nodes.get("USD_NEWS_SHOCK").initializeLayerData(newsShockEmbed.forward(trainingData.get(Math.max(0,i-72)).getRegimeLabel()));

        }


        return learningList;

    }


    public void learn(List<MarketData> trainingData) {

        for(int i = 0; i < trainingData.size(); i++) {
            //System.out.println("Encoding attention for sequence " + i);
            for (TsetlinVertex vertex : nodes.values()) {
                graph.encodeAttentionAtVertex(i, vertex);
            }
            graph.addLabel(trainingData.get(i).getRegimeLabel());
        }

        for(int i = 0; i < 100; i++) {

            int nClauses = 200 + (int)(random.nextDouble() * 600); // Randomize number of clauses between 100 and 150 for each iteration
            int threshold = (int) (nClauses / 3.0); // Standard threshold is half the number of clauses
            float maxSpecificity = 8.5f + random.nextFloat() * 12.0f; // Randomize max specificity between 2.0 and 5.0
            int maxLiterals = (int) (random.nextFloat() * 310); // Randomize max literals between 5 and 15

            //print out the parameters for this iteration
            System.out.println(String.format("Iteration %d: nClauses=%d, threshold=%d, maxSpecificity=%.2f, maxLiterals=%d",
                    i+1, nClauses, threshold, maxSpecificity, maxLiterals));

            graph.buildAttention(nClauses, threshold, maxSpecificity, maxLiterals);

        }
    }


    public static void main(String[] args) {
        ForexRegimeGraphManager manager = new ForexRegimeGraphManager();

        //  Load the assembled chronological dataset
        List<MarketData> marketDataList;
        try {
            String path = "data/assembled_dataset.csv";

            marketDataList = DatasetReader.loadDataset(path);
            System.out.println("Loaded " + marketDataList.size() + " rows for visualization.");
        } catch (IOException e) {
            System.err.println("Critical Error loading CSV file: " + e.getMessage());
            return;
        }

        computeLabeledDataset(marketDataList);


        List<MarketData> trainSet = manager.addSample(marketDataList);
        manager.learn(trainSet);
    }



}