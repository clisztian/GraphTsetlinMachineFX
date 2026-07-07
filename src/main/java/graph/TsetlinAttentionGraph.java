package graph;


import encoding.embedding.IntervalEmbedding;
import encoding.embedding.LinearEmbedding;
import encoding.hypervector.SparseBinaryVector;
import tsetlin.GraphAttentionLearning;
import tsetlin.GraphEncoder;
import tsetlin.TsetlinAttentionLearning;
import util.BitUtils;
import util.HV;
import util.MutableInt;

import java.util.*;
import java.util.logging.Logger;
import java.util.stream.LongStream;

public class TsetlinAttentionGraph extends DigraphEdgeList<TsetlinVertex, TsetlinEdge> {

    //logger
    final Logger logger = Logger.getLogger(TsetlinAttentionGraph.class.getName());

    private final int INT_SIZE = 32;
    private final int numberLayers;
    //labels for the graph, must equal the number of data points in each vertex

    //Tsetlin Graph
    private GraphAttentionLearning graphAttentionLearning;
    private TsetlinAttentionLearning tsetlinAttentionLearning;

    private IntervalEmbedding nodeEmbedding;
    private LinearEmbedding edgeEmbedding;
    private SparseBinaryVector encodedGraph;



    //data sets for learning graph
    private int[][][] layeredX;
    private final List<Integer> labels = new ArrayList<>();
    private int nClasses;
    private int nClauses;
    private int nodeChunks;

    private GraphEncoder encoder;

    private final Map<Vertex<TsetlinVertex>, Double> pageRankMap = new HashMap<>();

    private final Map<SparseBinaryVector, TsetlinEdge> keyEmbeddings = new HashMap<>();



    /**
     * Instantiates a new Tsetlin graph.
     * All vertices and edges need to have a name and an importance value.
     */
    public TsetlinAttentionGraph(int numberLayers) {
        super();

        this.numberLayers = numberLayers;
        //instantiates the vertices and edges using a TreeMap to ensure sorted order with respect to the importance
        this.vertices = new TreeMap<>((v1, v2) -> Float.compare(v2.getVertexImportance(), v1.getVertexImportance()));
        this.edges = new TreeMap<>((e1, e2) -> Float.compare(e2.getEdgeImportance(), e1.getEdgeImportance()));

    }

    /**
     * Get the keys of the vertices in the graph as a List in order of TreeMap sorting
     */
    public List<TsetlinVertex> getVertexKeys() {
        return new ArrayList<>(vertices.keySet());
    }

    /**
     * Vertex keys now have to be resorted after modifying vertex importance or adding new vertices.
     */
    public List<TsetlinVertex> getVerticesByImportance() {
        List<TsetlinVertex> sortedVertices = new ArrayList<>(vertices.keySet());
        sortedVertices.sort(Comparator.comparing(TsetlinVertex::getVertexImportance).reversed());
        return sortedVertices;
    }


    /**
     * Returns the index-th data of the graph.
     * Will be returned in order of importance of the vertices (order added, PageRank, etc.)
     * @param index
     * @return
     */
    public List<SparseBinaryVector> getVertexDataset(int index) {

        List<SparseBinaryVector> vertexData = new ArrayList<>();

        for (TsetlinVertex vertex : vertices.keySet()) {

            // Check if the index is within the bounds of the vertex data other throw exception
            if (index < 0 || index >= vertex.getVertexData().size()) {
                throw new IndexOutOfBoundsException("Index out of bounds for vertex data");
            }

            if (index < vertex.getVertexData().size()) {
                vertexData.add(vertex.getVertexData().get(index).getLayerVector(0));
            }
        }
        return vertexData;
    }

    /**
     * Encodes the attention at the vertex for the index-th data state
     * The attention is the
     * Vertex data (layer 1), vertex bind message data (layer 2), vertex bind message data (layer 3)
     *
     */
    public void encodeAttentionAtVertex(int index, TsetlinVertex vertex) {

        //if index is out of bounds, throw exception
        if (index < 0 || index >= vertex.getVertexData().size()) {
            throw new IndexOutOfBoundsException("Index out of bounds for vertex data");
        }

        VertexAttention vertexAttention = vertex.getVertexData().get(index);

        //check that exactly one layer of data is present, namely the size of the layerVectors should be 1
        if (vertexAttention.getLayerVectors().size() != 1) {
            throw new IllegalStateException("Vertex attention should have exactly one layer of data");
        }

        //order of edges doesn't matter here, get all incoming messages
        List<Edge<TsetlinEdge,TsetlinVertex>> edges = List.copyOf(incidentEdges(vertices.get(vertex)));

        List<SparseBinaryVector> bundledAttention = new ArrayList<>();
        //add the zero vector as the first element to the bundled attention
        bundledAttention.add(SparseBinaryVector.getZeroVector());

        //with incoming edges, get the vertex data and the edge data
        for (Edge<TsetlinEdge,TsetlinVertex> edge : edges) {

            TsetlinEdge tsetlinEdge = edge.element();
            TsetlinVertex tsetlinVertex = edge.vertices()[0].element();

            SparseBinaryVector layerAttention = tsetlinVertex.getVertexData().get(index).getLayerVector(0).bind(tsetlinEdge.getEdgeEncoder());
            bundledAttention.add(layerAttention);
        }

        //bundle the attention data
        SparseBinaryVector bundledAttentionVector = SparseBinaryVector.bundle(bundledAttention);

        vertexAttention.addLayerVector(bundledAttentionVector);

    }







    public void addLabel(int label) {
        labels.add(label);
    }

    public void addLabels(List<Integer> labels) {
        this.labels.addAll(labels);
    }

    public List<Integer> getLabels() {
        return labels;
    }

    public int getNumberLayers() {
        return numberLayers;
    }

    /**
     * After all vertices, edges, and data added, builds the Tsetlin attention graph
     */
    public void buildTsetlinAttention(int nClauses, int threshold, float maxSpecificity, int maxLiterals, float dropout) {

        encoder = new GraphEncoder(numVertices(), HV.DIMENSION, HV.DIMENSION, numberLayers);

        //get number of distinct classes by computing distinct labels
        nClasses = (int) labels.stream().distinct().count();

        //initialize the graph attention learning
        graphAttentionLearning = new GraphAttentionLearning(encoder, nClauses, nClasses, threshold, maxSpecificity, true, dropout);
        graphAttentionLearning.setMaxNumberOfLiterals(maxLiterals);

        //build data sets from the vertices
        layeredX = new int[labels.size()][numberLayers][];
        for(int i = 0; i < labels.size(); i++) {

            List<int[]> featureVectorLayerOne = new ArrayList<>();
            List<int[]> featureVectorLayerTwo = new ArrayList<>();
            for (TsetlinVertex vertex : vertices.keySet()) {
                //get the data set for the vertex

                VertexAttention sample = vertex.getVertexData().get(i);
                featureVectorLayerOne.add(BitUtils.longsToInts(sample.getLayerVector(0).encoded()));
                featureVectorLayerTwo.add(BitUtils.longsToInts(sample.getLayerVector(1).encoded()));
            }

            //concatenate all arrays into one per layer
            int[] layerOne = new int[featureVectorLayerOne.size() * featureVectorLayerOne.get(0).length];
            int[] layerTwo = new int[featureVectorLayerTwo.size() * featureVectorLayerTwo.get(0).length];

            int featureLength = featureVectorLayerOne.get(0).length;
            for (int j = 0; j < featureVectorLayerOne.size(); j++) {
                System.arraycopy(featureVectorLayerOne.get(j), 0, layerOne, j * featureLength, featureLength);
                System.arraycopy(featureVectorLayerTwo.get(j), 0, layerTwo, j * featureLength, featureLength);
            }

            //add the layer to the layeredX
            layeredX[i][0] = layerOne;
            //layeredX[i][1] = layerTwo;

        }

    }


    public void buildGraphAttention(int nClauses, int threshold, float maxSpecificity, int maxLiterals) {

        //build train/test data sets
        Random random = new Random();

        encoder = new GraphEncoder(numVertices(), HV.DIMENSION, HV.DIMENSION, numberLayers);

        //get number of distinct classes by computing distinct labels
        nClasses = (int) labels.stream().distinct().count();
        this.nClauses = nClauses;
        this.nodeChunks = (numVertices() - 1) / INT_SIZE + 1;

        //initialize the graph attention learning
        tsetlinAttentionLearning = new TsetlinAttentionLearning(encoder, nClauses, nClasses, threshold, maxSpecificity, true, maxLiterals);


        //build data sets from the vertices
        layeredX = new int[labels.size()][numberLayers][];
        int[] y = new int[labels.size()];

        for(int i = 0; i < labels.size(); i++) {

            int[] vertexLayerData = new int[numVertices() * encoder.getLayer_ta_chunks()];

            int vertexCount = 0;
            for (TsetlinVertex vertex : vertices.keySet()) {
                //get the data set for the vertex

                VertexAttention sample = vertex.getVertexData().get(i);
                int[] vertexMessage = BitUtils.longsToInts(sample.getLayerVector(0).encoded());


                System.arraycopy(vertexMessage, 0, vertexLayerData, vertexCount * encoder.getLayer_ta_chunks(), vertexMessage.length);
                vertexCount++;
            }

            //add the layer to the layeredX
            layeredX[i][0] = vertexLayerData;
            y[i] = labels.get(i);

        }




        //take 80 percent of the data for training
        int trainSize = (int)(layeredX.length * .8);
        //build a random set of unique indices from 1 - X.length
        int[] indices = new int[layeredX.length];
        for(int i = 0; i < layeredX.length; i++) {
            indices[i] = i;
        }
        //shuffle the indices
        for(int i = 0; i < layeredX.length; i++) {
            int j = (int)(random.nextDouble() * layeredX.length);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        int[][][] X_train = new int[trainSize][][];
        int[] y_train = new int[trainSize];
        for(int i = 0; i < trainSize; i++) {
            X_train[i] = layeredX[indices[i]];
            y_train[i] = y[indices[i]];
        }

        int[][][] X_test = new int[layeredX.length - trainSize][][];
        int[] y_test = new int[layeredX.length - trainSize];

        for(int i = trainSize; i < layeredX.length; i++) {
            X_test[i - trainSize] = layeredX[indices[i]];
            y_test[i - trainSize] = y[indices[i]];
        }





        for(int epoch = 0; epoch < 4; epoch++) {
//            tsetlinAttentionLearning.fitOneLyer(encoding.getX_train(), encoding.getY_train());
//            tsetlinAttentionLearning.predict(encoding.getX_test(), encoding.getY_test());

            tsetlinAttentionLearning.fitOneLyer(X_train, y_train);
            tsetlinAttentionLearning.predict(X_test, y_test);
        }


    }




    public void buildAttention(int nClauses, int threshold, float maxSpecificity, int maxLiterals) {

        //build train/test data sets
        Random random = new Random();

        encoder = new GraphEncoder(numVertices(), HV.DIMENSION, HV.DIMENSION, numberLayers);

        //get number of distinct classes by computing distinct labels
        nClasses = (int) labels.stream().distinct().count();
        this.nClauses = nClauses;
        this.nodeChunks = (numVertices() - 1) / INT_SIZE + 1;

        //initialize the graph attention learning
        tsetlinAttentionLearning = new TsetlinAttentionLearning(encoder, nClauses, nClasses, threshold, maxSpecificity, true, maxLiterals);


        //build data sets from the vertices
        layeredX = new int[labels.size()][numberLayers][];
        int[] y = new int[labels.size()];

        for(int i = 0; i < labels.size(); i++) {

            int[] vertexLayerData = new int[numVertices() * encoder.getLayer_ta_chunks()];

            int vertexCount = 0;
            for (TsetlinVertex vertex : vertices.keySet()) {
                //get the data set for the vertex

                VertexAttention sample = vertex.getVertexData().get(i);
                int[] vertexMessage = BitUtils.longsToInts(sample.getLayerVector(0).encoded());


                System.arraycopy(vertexMessage, 0, vertexLayerData, vertexCount * encoder.getLayer_ta_chunks(), vertexMessage.length);
                vertexCount++;
            }

            //add the layer to the layeredX
            layeredX[i][0] = vertexLayerData;
            y[i] = labels.get(i);

        }

        for(int epoch = 0; epoch < 2; epoch++) {
            fit(layeredX, y);
        }



    }










    public void fit(int[][][] X, int[] y) {

        Random random = new Random(4);

        //create one-hot encoding for y
        int[][] y_one_hot = new int[y.length][nClasses];
        for(int i = 0; i < y.length; i++) {
            y_one_hot[i][y[i]] = 1;
        }


        List<Integer> testIndices = new ArrayList<>();

        int correct = 0;
        int count = 0;



        for(int k = 0; k < X.length; k++) {

            int[][] globalClauseNodeOutput = new int[nClauses][nodeChunks];
            int[][] globalClauseNodeOutputLayer = new int[nClauses][nodeChunks];

            int index = random.nextInt(X.length);

            if(random.nextFloat() < .20  ) {
                testIndices.add(index);
                continue; //skip this index for training, it will be used for testing
            }

            //only allow y[index] = 0 only 10% of the time
            if(y[index] == 0 && random.nextFloat() < .93) {
                continue;
            }

            //first need to evaluate to get the output clause vertex map and the class_sum output
            int[] numberOfIncludeActions = tsetlinAttentionLearning.evaluateVertexLayer(X[index][0], globalClauseNodeOutput);


            //now build the layer two data set
            int[] vertexLayerData = new int[numVertices() * encoder.getLayer_ta_chunks()];
            //now build second layer of data given the global clause node output
            int vertexCount = 0;
            //go through all vertices to build messages
            for (TsetlinVertex vertex : vertices.keySet()) {

                List<SparseBinaryVector> bundledAttention = new ArrayList<>();
                bundledAttention.add(SparseBinaryVector.getZeroVector());

                //for current vertex, if any clause in globalClauseNodeOutput is one, then add the vertex data
                for(int i = 0; i < nClauses; i++) {

                    int source_node_chunk = vertexCount / INT_SIZE;
                    int source_node_pos = vertexCount % INT_SIZE;

                    if ((globalClauseNodeOutput[i][source_node_chunk] & (1 << source_node_pos)) > 0) {
                        VertexAttention sample = vertex.getVertexData().get(index);
                        bundledAttention.add(sample.getLayerVector(1));
                        break;
                    }
                }

                int[] vertexMessage = BitUtils.longsToInts(SparseBinaryVector.bundle(bundledAttention).encoded());

                System.arraycopy(vertexMessage, 0, vertexLayerData, vertexCount * encoder.getLayer_ta_chunks(), vertexMessage.length);

                vertexCount++;
            }

            X[index][1] = vertexLayerData;

            tsetlinAttentionLearning.evaluateLayer(vertexLayerData, globalClauseNodeOutput, globalClauseNodeOutputLayer, numberOfIncludeActions, 1);

            int[] class_sums = tsetlinAttentionLearning.evaluate(globalClauseNodeOutputLayer);

            int predictedLabel = -1;

            int maxClassSum = -Integer.MAX_VALUE;
            for(int i = 0; i < class_sums.length; i++) {
                if(class_sums[i] > maxClassSum) {
                    maxClassSum = class_sums[i];
                    predictedLabel = i;
                }
            }

            correct += (predictedLabel == y[index]) ? 1 : 0;

            //after evaluating the layer, update the graph
            tsetlinAttentionLearning.update(X[index], y_one_hot[index], class_sums, globalClauseNodeOutputLayer, numberOfIncludeActions);

            count++;
        }
        //compute the accuracy
        System.out.println("Accuracy: " + ((float) correct / (float)count));



        //now test the model on the test indices
        correct = 0;
        count = 0;

        //compute accruacy for each class
        Map<Integer, MutableInt> classCorrect = new HashMap<>();
        Map<Integer, MutableInt> classTotal = new HashMap<>();



        //add 0 counts for each class 0-3
        for(int i = 0; i < nClasses; i++) {
            classCorrect.put(i, new MutableInt(0));
            classTotal.put(i, new MutableInt(0));
        }

        //confusion matrix
        Map<Integer, Map<Integer, MutableInt>> confusionMatrix = new HashMap<>();

        //initialize confusion matrix
        for(int i = 0; i < nClasses; i++) {
            confusionMatrix.put(i, new HashMap<>());
            for(int j = 0; j < nClasses; j++) {
                confusionMatrix.get(i).put(j, new MutableInt(0));
            }
        }


        for(int index : testIndices) {

            int[][] globalClauseNodeOutput = new int[nClauses][nodeChunks];
            int[][] globalClauseNodeOutputLayer = new int[nClauses][nodeChunks];

            //first need to evaluate to get the output clause vertex map and the class_sum output
            int[] numberOfIncludeActions = tsetlinAttentionLearning.evaluateVertexLayer(X[index][0], globalClauseNodeOutput);


            //now build the layer two data set
            int[] vertexLayerData = new int[numVertices() * encoder.getLayer_ta_chunks()];
            //now build second layer of data given the global clause node output
            int vertexCount = 0;
            //go through all vertices to build messages
            for (TsetlinVertex vertex : vertices.keySet()) {

                List<SparseBinaryVector> bundledAttention = new ArrayList<>();
                bundledAttention.add(SparseBinaryVector.getZeroVector());

                //for current vertex, if any clause in globalClauseNodeOutput is one, then add the vertex data
                for(int i = 0; i < nClauses; i++) {

                    int source_node_chunk = vertexCount / INT_SIZE;
                    int source_node_pos = vertexCount % INT_SIZE;

                    if ((globalClauseNodeOutput[i][source_node_chunk] & (1 << source_node_pos)) > 0) {
                        VertexAttention sample = vertex.getVertexData().get(index);
                        bundledAttention.add(sample.getLayerVector(1));
                        break;
                    }
                }

                int[] vertexMessage = BitUtils.longsToInts(SparseBinaryVector.bundle(bundledAttention).encoded());

                System.arraycopy(vertexMessage, 0, vertexLayerData, vertexCount * encoder.getLayer_ta_chunks(), vertexMessage.length);

                vertexCount++;
            }

            X[index][1] = vertexLayerData;

            tsetlinAttentionLearning.evaluateLayer(vertexLayerData, globalClauseNodeOutput, globalClauseNodeOutputLayer, numberOfIncludeActions, 1);

            int[] class_sums = tsetlinAttentionLearning.evaluate(globalClauseNodeOutputLayer);

            //print class_sums
            int predictedLabel = -1;

            int maxClassSum = -Integer.MAX_VALUE;
            for(int i = 0; i < class_sums.length; i++) {
                if(class_sums[i] > maxClassSum) {
                    maxClassSum = class_sums[i];
                    predictedLabel = i;
                }
            }

            correct += (predictedLabel == y[index]) ? 1 : 0;

            //update class counts
            classTotal.get(y[index]).increment();
            if(predictedLabel == y[index]) {
                classCorrect.get(y[index]).increment();
            }

            //update confusion matrix
            confusionMatrix.get(y[index]).get(predictedLabel).increment();


            count++;
        }

        //compute the accuracy
        System.out.println("Test Accuracy: " + ((float) correct / (float)count));

        //print class accuracies
        for(int i = 0; i < nClasses; i++) {
            int correctCount = classCorrect.get(i).get();
            int totalCount = classTotal.get(i).get();
            float classAccuracy = (totalCount > 0) ? (float) correctCount / (float) totalCount : 0;
            System.out.println("Class " + i + " Accuracy: " + classAccuracy);
        }

        //print confusion matrix
        System.out.println("Confusion Matrix:");
        for(int i = 0; i < nClasses; i++) {
            for(int j = 0; j < nClasses; j++) {
                System.out.print(confusionMatrix.get(i).get(j).get() + " ");
            }
            System.out.println();
        }

        //confusion matrix statistics
        for(int i = 0; i < nClasses; i++) {
            int truePositive = confusionMatrix.get(i).get(i).get();
            int falsePositive = 0;
            int falseNegative = 0;
            for(int j = 0; j < nClasses; j++) {
                if(j != i) {
                    falsePositive += confusionMatrix.get(j).get(i).get();
                    falseNegative += confusionMatrix.get(i).get(j).get();
                }
            }
            float precision = (truePositive + falsePositive > 0) ? (float) truePositive / (float) (truePositive + falsePositive) : 0;
            float recall = (truePositive + falseNegative > 0) ? (float) truePositive / (float) (truePositive + falseNegative) : 0;
            float f1Score = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0;
            System.out.println("Class " + i + " Precision: " + precision + " Recall: " + recall + " F1 Score: " + f1Score);
        }





    }



    public void fitOneLayer(int[][][] X, int[] y) {

        //create one-hot encoding for y
        int[][] y_one_hot = new int[y.length][nClasses];
        for(int i = 0; i < y.length; i++) {
            y_one_hot[i][y[i]] = 1;
        }

        int[][] globalClauseNodeOutput = new int[nClauses][nodeChunks];


        int correct = 0;
        for(int index = 0; index < X.length; index++) {


            //first need to evaluate to get the output clause vertex map and the class_sum output
            int[] numberOfIncludeActions = tsetlinAttentionLearning.evaluateVertexLayer(X[index][0], globalClauseNodeOutput);

            int[] class_sums = tsetlinAttentionLearning.evaluate(globalClauseNodeOutput);

            tsetlinAttentionLearning.update(X[index], y_one_hot[index], class_sums, globalClauseNodeOutput, numberOfIncludeActions);
            //print class_sums
            //does the larger of class_sums match the label?
            int predictedLabel = -1;

            int maxClassSum = -Integer.MAX_VALUE;
            for(int i = 0; i < class_sums.length; i++) {
                if(class_sums[i] > maxClassSum) {
                    maxClassSum = class_sums[i];
                    predictedLabel = i;
                }
            }

            correct += (predictedLabel == y[index]) ? 1 : 0;

            //System.out.println(class_sums[0] + " " + class_sums[1] + " " + predictedLabel + " " + y[index]);

        }
        //compute the accuracy
        System.out.println("Accuracy: " + ((float) correct / (float)X.length));
    }

    public void predict(int[][][] X, int[] y) {

        //create one-hot encoding for y
        int[][] y_one_hot = new int[y.length][nClasses];
        for(int i = 0; i < y.length; i++) {
            y_one_hot[i][y[i]] = 1;
        }

        int[][] globalClauseNodeOutput = new int[nClauses][nodeChunks];

        int correct = 0;
        for(int index = 0; index < X.length; index++) {

            //first need to evaluate to get the output clause vertex map and the class_sum output
            int[] numberOfIncludeActions = tsetlinAttentionLearning.evaluateVertexLayer(X[index][0], globalClauseNodeOutput);
            int[] class_sums = tsetlinAttentionLearning.evaluate(globalClauseNodeOutput);

            //does the larger of class_sums match the label?
            int predictedLabel = -1;

            int maxClassSum = -Integer.MAX_VALUE;
            for(int i = 0; i < class_sums.length; i++) {
                if(class_sums[i] > maxClassSum) {
                    maxClassSum = class_sums[i];
                    predictedLabel = i;
                }
            }

            correct += (predictedLabel == y[index]) ? 1 : 0;

        }
        //compute the accuracy

        System.out.println("Predict Accuracy: " + ((float) correct / (float)X.length));
    }





    /**
     * Updates the index -th data of the vertex with the attention data
     */
    public void fitGraph(int[][][] X, int[] y) {

        int correct = 0;

        for(int index = 0; index < X.length; index++) {

            int label = y[index];
            int[] Xi = X[index][0];
            int[] layerTwo = new int[X[index][0].length]; //layer two is the same size as layer one (number of features * number of vertices)

            //evaluates the vertex clause map
            int[][] layer1map = graphAttentionLearning.evaluateVertexClauseMap(Xi, graphAttentionLearning.getTa_atom_state());


            int vertexCount = 0;
            //go through all vertices to build messages
            for (TsetlinVertex vertex : vertices.keySet()) {

                List<SparseBinaryVector> bundledAttention = new ArrayList<>();
                bundledAttention.add(SparseBinaryVector.getZeroVector());

                //get the edge data
                List<Edge<TsetlinEdge, TsetlinVertex>> edges = List.copyOf(incidentEdges(vertices.get(vertex)));
                VertexAttention vertexAttention = vertex.getVertexData().get(index);
                //fitGraph the vertex with the attention data
                for (Edge<TsetlinEdge, TsetlinVertex> edge : edges) {

                    TsetlinEdge tsetlinEdge = edge.element();
                    TsetlinVertex tsetlinVertex = edge.vertices()[0].element();

                    //get the index of the vertex
                    int vertexIndex = vertices.keySet().stream().toList().indexOf(tsetlinVertex);

                    //if row layer1map[vertexCount] is all zeros skip
                    if(isAllZeros(layer1map[vertexIndex])) {
                        continue;
                    }

                    SparseBinaryVector layerAttention = tsetlinVertex.getVertexData().get(index).getLayerVector(0).bind(tsetlinEdge.getEdgeEncoder());
                    bundledAttention.add(layerAttention);
                }

                //bundle the attention data
                SparseBinaryVector bundledAttentionVector = SparseBinaryVector.bundle(bundledAttention);

                vertexAttention.setLayerVector(1, bundledAttentionVector);

                int[] vertexMessage = BitUtils.longsToInts(bundledAttentionVector.encoded());

                System.arraycopy(vertexMessage, 0, layerTwo, vertexCount * vertexMessage.length, vertexMessage.length);

                vertexCount++;
            }

            //set Xi layer two
            X[index][1] = layerTwo;

            int out = graphAttentionLearning.updateGraph(X[index], label);


            correct += (out == label) ? 1 : 0;
        }

        //compute the accuracy
        int accuracy = (int) ((float) correct / X.length * 100);
        System.out.println("Accuracy: " + ((float) correct / (float)X.length));
    }


    public void predictGraph(int[][][] X, int[] y) {

        int correct = 0;

        for(int index = 0; index < X.length; index++) {

            int label = y[index];
            int[] Xi = X[index][0];
            int[] layerTwo = new int[X[index][0].length]; //layer two is the same size as layer one (number of features * number of vertices)

            //evaluates the vertex clause map
            int[][] layer1map = graphAttentionLearning.evaluateVertexClauseMap(Xi, graphAttentionLearning.getTa_atom_state());


            int vertexCount = 0;
            //go through all vertices to build messages
            for (TsetlinVertex vertex : vertices.keySet()) {

                List<SparseBinaryVector> bundledAttention = new ArrayList<>();
                bundledAttention.add(SparseBinaryVector.getZeroVector());

                //get the edge data
                List<Edge<TsetlinEdge, TsetlinVertex>> edges = List.copyOf(incidentEdges(vertices.get(vertex)));
                VertexAttention vertexAttention = vertex.getVertexData().get(index);
                //fitGraph the vertex with the attention data
                for (Edge<TsetlinEdge, TsetlinVertex> edge : edges) {

                    TsetlinEdge tsetlinEdge = edge.element();
                    TsetlinVertex tsetlinVertex = edge.vertices()[0].element();

                    //get the index of the vertex
                    int vertexIndex = vertices.keySet().stream().toList().indexOf(tsetlinVertex);

                    //if row layer1map[vertexCount] is all zeros skip
                    if(isAllZeros(layer1map[vertexIndex])) {
                        continue;
                    }

                    SparseBinaryVector layerAttention = tsetlinVertex.getVertexData().get(index).getLayerVector(0).bind(tsetlinEdge.getEdgeEncoder());
                    bundledAttention.add(layerAttention);
                }

                //bundle the attention data
                SparseBinaryVector bundledAttentionVector = SparseBinaryVector.bundle(bundledAttention);

                vertexAttention.setLayerVector(1, bundledAttentionVector);

                int[] vertexMessage = BitUtils.longsToInts(bundledAttentionVector.encoded());

                System.arraycopy(vertexMessage, 0, layerTwo, vertexCount * vertexMessage.length, vertexMessage.length);

                vertexCount++;
            }

//            System.out.println("Layer 1 map:");
//            for (int j : layerTwo) {
//                System.out.print(Integer.toBinaryString(j) + " ");
//            }
//            System.out.println();


            //set Xi layer two
            X[index][1] = layerTwo;

            int out = graphAttentionLearning.predictGraph(X[index]);

            correct += (out == label) ? 1 : 0;
        }

        //compute the accuracy
        int accuracy = (int) ((float) correct / X.length * 100);

        System.out.println("Accuracy: " + ((float) correct / (float)X.length));
    }



    public void train() {

        //ensure layeredX is same length labels
        if (layeredX.length != labels.size()) {
            throw new IllegalStateException("LayeredX and labels must be the same length");
        }



        Random random = new Random();

        //take 80 percent of the data for training
        int trainSize = (int)(layeredX.length * .8);
        //build a random set of unique indices from 1 - X.length
        int[] indices = new int[layeredX.length];
        for(int i = 0; i < layeredX.length; i++) {
            indices[i] = i;
        }
        //shuffle the indices
        for(int i = 0; i < layeredX.length; i++) {
            int j = (int)(random.nextDouble() * layeredX.length);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        int[][][] X_train = new int[trainSize][][];
        int[] y_train = new int[trainSize];
        for(int i = 0; i < trainSize; i++) {
            X_train[i] = layeredX[indices[i]];
            y_train[i] = labels.get(indices[i]);
        }

        int[][][] X_test = new int[layeredX.length - trainSize][][];
        int[] y_test = new int[layeredX.length - trainSize];

        for(int i = trainSize; i < layeredX.length; i++) {
            X_test[i - trainSize] = layeredX[indices[i]];
            y_test[i - trainSize] = labels.get(indices[i]);
        }

        for(int epochs = 0; epochs < 8; epochs++) {
            fitGraph(X_train, y_train);
            predictGraph(X_test, y_test);
        }

    }


    /**
     * If node importance and edge are defined, we create linear embeddings for the nodes and edges.
     * @param maxNodeEmbedding
     * @param minEdgeEmbedding
     * @param maxEdgeEmbedding
     */
    public void buildGraphEmbeddings(float minImportance, float maxNodeEmbedding, float minEdgeEmbedding, float maxEdgeEmbedding) {
        nodeEmbedding = new IntervalEmbedding(minImportance, maxNodeEmbedding, numVertices());
        edgeEmbedding = new LinearEmbedding(minEdgeEmbedding, maxEdgeEmbedding, 10);
    }

    /**
     * Returns the embedding for the vertex.
     * @param vertex
     * @return
     */
    public SparseBinaryVector getEmbedding(TsetlinVertex vertex) {

        if (nodeEmbedding == null) {
            return SparseBinaryVector.getZeroVector();
        }
        return nodeEmbedding.forward((double)vertex.getVertexImportance());

    }

    /**
     * Returns the embedding for the edge.
     * @param edge
     * @return
     */
    public SparseBinaryVector getEmbedding(TsetlinEdge edge) {

        if (edgeEmbedding == null) {
            return SparseBinaryVector.getZeroVector();
        }
        return edgeEmbedding.forward((double)edge.getEdgeImportance());
    }



    /**
     * Returns the graph attention learning object
     * @return
     */
    public final GraphAttentionLearning getGraphAttentionLearning() {
        return graphAttentionLearning;
    }

    /**
     * Number of layers x number of data points x number of features
     * @return
     */
    public int[][][] getLayeredX() {
        return layeredX;
    }

    public static boolean isAllZeros(int[] v) {
        for (int i : v) {
            if (i != 0) {
                return false;
            }
        }
        return true;
    }


    /**
     * Returns the bitwise AND of two 2D arrays
     * @param a
     * @param b
     * @return
     */
    public static int[][] and(int[][] a, int[][] b) {
        int[][] result = new int[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                result[i][j] = (a[i][j] & b[i][j]) == b[i][j] ? 1 : 0;
            }
        }
        return result;
    }


    public TsetlinVertex findMostInfluentialNode(SparseBinaryVector recovered, int sampleIndex) {

        TsetlinVertex closestVertex = null;
        int minDistance = Integer.MAX_VALUE;

        for (TsetlinVertex vertex : this.getVertexKeys()) {
            SparseBinaryVector candidate = vertex.getVertexData(sampleIndex).getLayerVector(0);

            if (candidate == null) {
                continue; // Skip if vertex has no encoding for this sample
            }

            int distance = recovered.hammingDistance(candidate);
            if (distance < minDistance) {
                minDistance = distance;
                closestVertex = vertex;
                //logger.info("Found closer vertex: " + vertex.getName() + " with distance: " + distance);
            }
        }

        logger.info("Closest vertex to recovered vector: " + closestVertex.getName() + " with distance: " + minDistance);

        return closestVertex;
    }


    public TsetlinEdge findMostInfluentialEdge(int[][] literalCounts) {
        if (keyEmbeddings.isEmpty()) {
            throw new IllegalStateException("Edge map is not initialized.");
        }

        // 1. Compute the average number of bits ON in edgeMap keys
        int totalBitsOn = 0;

        for (SparseBinaryVector vec : keyEmbeddings.keySet()) {
            totalBitsOn += (int) LongStream.of(vec.getSegments()).map(Long::bitCount).sum();
        }

        int avgBitsOn = (int)Math.max(1, (double)totalBitsOn / keyEmbeddings.size());  // ensure at least 1 bit

        logger.info("Average bits ON in edgeMap keys: " + avgBitsOn);

        // 2. Flatten literalCounts (2 rows: [0] positive literals, [1] negative literals) into one array
        int[] flattened = new int[literalCounts[0].length];
        for (int i = 0; i < flattened.length; i++) {
            flattened[i] = literalCounts[0][i] + literalCounts[1][i];
        }

        // 3. Find top-N bit positions by count, create new SparseBinaryVector
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < flattened.length; i++) indices.add(i);
        indices.sort((a, b) -> Integer.compare(flattened[b], flattened[a]));

        long[] segments = new long[HV.SEGMENT_SIZE];
        for (int i = 0; i < avgBitsOn && i < indices.size(); i++) {
            int pos = indices.get(i);
            int seg = pos / HV.LONG_SIZE;
            int bit = pos % HV.LONG_SIZE;
            segments[seg] |= (1L << bit);
        }

        SparseBinaryVector recovered = new SparseBinaryVector(segments);

        // 4. Find the closest vector in edgeMap using Hamming distance
        int bestDist = Integer.MAX_VALUE;
        TsetlinEdge closestEdge = null;

        for (Map.Entry<SparseBinaryVector, TsetlinEdge> entry : keyEmbeddings.entrySet()) {
            int dist = recovered.hammingDistance(entry.getKey());
            if (dist < bestDist) {
                bestDist = dist;
                closestEdge = entry.getValue();
            }
        }

        logger.info("Closest edge to recovered vector with distance: " + bestDist);

        return closestEdge;
    }

    public List<Map.Entry<TsetlinEdge, Double>> findTopInfluentialEdges(int[][] literalCounts, int topE) {
        // Step 1: Compute average number of bits ON in key encodings
        int totalBits = 0;
        for (SparseBinaryVector vec : keyEmbeddings.keySet()) {
            totalBits += Long.bitCount(Arrays.stream(vec.getSegments()).reduce(0L, (a, b) -> a + Long.bitCount(b)));
        }
        int avgBits = totalBits / keyEmbeddings.size();

        // Step 2: Flatten literalCounts and find top-k indices
        int[] literalSum = new int[literalCounts[0].length];
        for (int[] literalSet : literalCounts) {
            for (int i = 0; i < literalSet.length; i++) {
                literalSum[i] += literalSet[i];
            }
        }


        // 3. Find top-N bit positions by count, create new SparseBinaryVector
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < literalSum.length; i++) indices.add(i);
        indices.sort((a, b) -> Integer.compare(literalSum[b], literalSum[a]));



        long[] segments = new long[HV.SEGMENT_SIZE];
        for (int i = 0; i < avgBits && i < indices.size(); i++) {
            int pos = indices.get(i);
            int seg = pos / HV.LONG_SIZE;
            int bit = pos % HV.LONG_SIZE;
            segments[seg] |= (1L << bit);
        }

        SparseBinaryVector recovered = new SparseBinaryVector(segments);

        // Step 4: Find top E matches with minimal Hamming distance
        PriorityQueue<Map.Entry<TsetlinEdge, Double>> pq = new PriorityQueue<>(
                Comparator.comparingDouble(Map.Entry::getValue)
        );

        int edgeCount = 0;
        for (Map.Entry<SparseBinaryVector, TsetlinEdge> entry : keyEmbeddings.entrySet()) {
            SparseBinaryVector vec = entry.getKey();
            double distance = recovered.hammingDistance(vec);
            pq.add(Map.entry(entry.getValue(), distance));
            //logger.info(edgeCount + " Edge: " + entry.getValue().getEdgeName() + ", Distance: " + distance);
            edgeCount++;
        }

        List<Map.Entry<TsetlinEdge, Double>> top = new ArrayList<>();
        for (int i = 0; i < topE && !pq.isEmpty(); i++) {
            top.add(pq.poll());
        }

        return top;
    }



    //static main
    public static void main(String[] args) {
        // Example usage
        TsetlinAttentionGraph graph = new TsetlinAttentionGraph(1);
        // Add vertices, edges, and data to the graph here
        // ...

        // Build the Tsetlin attention graph
        graph.buildTsetlinAttention(100, 10, 0.5f, 5, 0.1f);

        // Train the graph
        graph.train();

        // Predict using the graph
        int[][][] testX = new int[10][2][];
        int[] testY = new int[10];
        graph.predict(testX, testY);
    }


    public SparseBinaryVector getEncodedGraph() {
        return encodedGraph;
    }

    public void setEncodedGraph(SparseBinaryVector encodedGraph) {
        this.encodedGraph = encodedGraph;
    }

    public Map<Vertex<TsetlinVertex>, Double> getPageRankMap() {
        return pageRankMap;
    }

    //set the pageRank map
    public void setPageRankMap(Map<Vertex<TsetlinVertex>, Double> pageRankMap) {
        this.pageRankMap.putAll(pageRankMap);
    }

    public Map<SparseBinaryVector, TsetlinEdge> getKeyEmbeddings() {
        return keyEmbeddings;
    }

    public void setEdgeMap(Map<SparseBinaryVector, TsetlinEdge> edgeMap) {
        keyEmbeddings.putAll(edgeMap);
    }

    public void mergeGraph(TsetlinAttentionGraph newGraph) {

        //add the vertices and edges from the new graph
        newGraph.getVertexKeys().forEach(this::addVertex);
        newGraph.edges().forEach(e -> insertEdge(e.vertices()[0], e.vertices()[1], e.element()));

    }

    private void addVertex(TsetlinVertex vertex) {
        if (!vertices.containsKey(vertex)) {
            Vertex<TsetlinVertex> v = addNewVertex(vertex);
            vertices.put(vertex, v);
        }
    }

    private Vertex<TsetlinVertex> addNewVertex(TsetlinVertex vertex) {
        return insertVertex(vertex);
    }
}
