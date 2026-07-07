package examples.sequence;

import containers.SmartGraphDemoContainer;
import encoding.hypervector.SparseBinaryVector;
import graph.TsetlinAttentionGraph;
import graph.TsetlinEdge;
import graph.TsetlinVertex;
import graphview.SmartCircularSortedPlacementStrategy;
import graphview.SmartGraphPanel;
import graphview.SmartGraphProperties;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;
import javafx.stage.StageStyle;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A simple sequence example to demonstrate the use of Tsetlin graph.
 */
public class SimpleSequenceExample extends Application {

    public static void main(String[] args) {
        launch(args);
    }

    final SparseBinaryVector aEncoder = SparseBinaryVector.randVector();
    final SparseBinaryVector bEncoder = SparseBinaryVector.randVector();


    Random rng = new Random();



    private List<String> generateStringSequences(int number, int length) {

        List<String> sequences = new ArrayList<>();

        List<String> tripleAsequences = new ArrayList<>();

        //generate number of sequences of length 4 of A and B randomly
        for (int i = 0; i < number; i++) {
            StringBuilder sequence = new StringBuilder();
            for (int j = 0; j < length; j++) {
                if (Math.random() > 0.5) {
                    sequence.append("A");
                } else {
                    sequence.append("B");
                }
            }

            //if the sequence contains three consecutive A's, add it to the list
            if (sequence.toString().contains("AAA")) {
                tripleAsequences.add(sequence.toString());
            }
            else {
                //if the sequence does not contain three consecutive A's, add it to the list
                sequences.add(sequence.toString());
            }
        }

        //remove enough sequences in sequences list so that tripleAsequences and sequences have the same size
        while (sequences.size() > tripleAsequences.size()) {
            sequences.remove(rng.nextInt(sequences.size()));
        }
        //add the sequences with three consecutive A's to the list
        sequences.addAll(tripleAsequences);

        return sequences;
    }

    private void createGraph(int length, TsetlinAttentionGraph graph) {

        SparseBinaryVector rightEdgeEncoder = SparseBinaryVector.randVector();
        SparseBinaryVector leftEdgeEncoder = SparseBinaryVector.randVector();

        List<TsetlinVertex> vertices = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            TsetlinVertex vertex = new TsetlinVertex("Lag " + i, 0.1f * i);
            graph.insertVertex(vertex);
            vertices.add(vertex);
        }


        //now create edges in sequence
        for (int i = 0; i < length - 1; i++) {
            TsetlinEdge edge = new TsetlinEdge("Lag " + i + " -> Lag " + (i + 1), 0.1f * (i + 1), rightEdgeEncoder);
            graph.insertEdge(vertices.get(i), vertices.get(i + 1), edge);
        }
        //now going the opposite direction
        for (int i = length - 1; i > 0; i--) {
            TsetlinEdge edge = new TsetlinEdge("Lag " + i + " -> Lag " + (i - 1), 0.5f * (i +1), leftEdgeEncoder);
            graph.insertEdge(vertices.get(i), vertices.get(i - 1), edge);
        }

    }



    private List<Integer> addDataToGraph(List<TsetlinVertex> vertices, List<String> sequences) {
        // Add data to the graph
        //check that the number of vertices is equal to the length of the sequences (just check the first one)
        if (vertices.size() != sequences.get(0).length()) {
            throw new IllegalArgumentException("Number of vertices must be equal to the length of the sequences");
        }

        List<Integer> labels = new ArrayList<>();

        //if three consecutive A's appear in the sequence, add 1 to the label, otherwise 0
        for (String sequence : sequences) {

            int label = sequence.contains("AAA") ? 1 : 0;
            labels.add(label);

            //map the letter to the SparseBinaryVector
            for(int i = 0; i < sequence.length(); i++) {
                if (sequence.charAt(i) == 'A') {
                    vertices.get(i).initializeLayerData(aEncoder);
                } else {
                    vertices.get(i).initializeLayerData(bEncoder);
                }
            }
        }
        return labels;
    }





    @Override
    public void start(Stage ignored) throws Exception {

        int lengthGraph = 8;


        TsetlinAttentionGraph graph = new TsetlinAttentionGraph(2);
        createGraph(lengthGraph, graph);



        List<TsetlinVertex> vertices = graph.getVertexKeys();
        //List<Integer> labels = addSimpleNumericaData(vertices, 1000);

        List<Integer> labels = addDataToGraph(vertices, generateStringSequences(400, lengthGraph));
        System.out.println("Labels: " + labels.size());


        /* Only Java 15 allows for multi-line strings */
        String customProps = "edge.label = true" + "\n" + "edge.arrow = false";

        SmartGraphProperties properties = new SmartGraphProperties(customProps);

        SmartGraphPanel<TsetlinVertex, TsetlinEdge> graphView = new SmartGraphPanel<>(graph, properties, new SmartCircularSortedPlacementStrategy(), "css/smartgraph.css");

        Scene scene = new Scene(new SmartGraphDemoContainer(graphView), 1024, 768);

        Stage stage = new Stage(StageStyle.DECORATED);
        stage.setTitle("JavaFX SmartGraph City Distances");
        stage.setMinHeight(500);
        stage.setMinWidth(800);
        stage.setScene(scene);
        stage.show();

        graphView.init();


        System.out.println("Graph initialized with " + graph.numVertices() + " vertices and " + graph.numEdges() + " edges.");
        System.out.println("Graph view initialized.");
        System.out.println("building first layer attention...");

        System.out.println("Sizes: " + labels.size() + " " + graph.getVertexKeys().get(0).getVertexData().size());

        for(int i = 0; i < labels.size(); i++) {
            //System.out.println("Encoding attention for sequence " + i);
            for (TsetlinVertex vertex : vertices) {
                graph.encodeAttentionAtVertex(i, vertex);
            }
            graph.addLabel(labels.get(i));
        }

        int nClauses = 20;
        int threshold = 45;
        float maxSpecificity = 2f;
        int maxLiterals = 20;
        float dropout = 0.8f;


        graph.buildAttention(nClauses, threshold, maxSpecificity, maxLiterals);
        //graph.trainGraph();

    }
}
