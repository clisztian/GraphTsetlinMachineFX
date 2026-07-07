package tsetlin;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class TsetlinAttentionLearning {

    final int INT_SIZE = 32;
    private final int STATE_BITS = 8;

    private final float drop_clause_p;
    private final int nClauses;



    private final int nClasses;
    private final int threshold;
    private final float max_specificity;
    private final boolean boost;
    private final int state_bits = 8;
    private final int la_chunks;
    private final int nodeChunks;

    private int max_literals = 0;


    private int numberLayers; //number of layers in the graph



    private int number_of_vertices; //total vertices used for the graph
    private int number_of_ta_chunks;
    private int number_of_features;
    private Random rng = new Random(21);

    private int[][] ta_atom_state; //[CLAUSES][LA_CHUNKS*STATE_BITS];

    private int[][] clauseWeights;
    private List<int[][]> ta_layer_list = new ArrayList<>();

    private int filter;

    private float s = 1.0f;


    private boolean focused_negative_sampling = false;

    private GraphAutomataUpdates automata;




    public TsetlinAttentionLearning(GraphEncoder graphEncoder, int nClauses, int nClasses, int threshold, float max_specificity, boolean boost, int max_literals) {


        this.nClauses = nClauses;
        this.nClasses = nClasses;
        this.threshold = threshold;
        this.max_specificity = max_specificity;
        this.boost = boost;
        this.drop_clause_p = 0.1f; //default drop clause probability

        this.number_of_vertices = graphEncoder.getNumber_of_vertices();
        this.number_of_features = graphEncoder.getFeatureDimension();
        this.number_of_ta_chunks = graphEncoder.getLayer_ta_chunks();
        this.la_chunks = graphEncoder.getLayer_ta_chunks();
        this.numberLayers = graphEncoder.getNumberLayers();
        this.s = max_specificity;
        this.max_literals = max_literals;

        this.nodeChunks = (number_of_vertices - 1) / INT_SIZE + 1;

        //print out all the parameters
        System.out.println("nClauses: " + nClauses);
        System.out.println("nClasses: " + nClasses);
        System.out.println("threshold: " + threshold);
        System.out.println("max_specificity: " + max_specificity);
        System.out.println("boost: " + boost);
        System.out.println("drop_clause_p: " + drop_clause_p);
        System.out.println("number_of_vertices: " + number_of_vertices);
        System.out.println("number_of_ta_chunks: " + number_of_ta_chunks);
        System.out.println("number_of_features: " + number_of_features);
        System.out.println("la_chunks: " + la_chunks);
        System.out.println("state_bits: " + state_bits);



        //initialize the automata

        if (((graphEncoder.getFeatureDimension()*2) % INT_SIZE) != 0) {
            this.filter  = (~(0xffffffff << ((graphEncoder.getFeatureDimension()*2) % INT_SIZE)));
        } else {
            this.filter = 0xffffffff;
        }

        ta_atom_state = initializeAtom();

        //initialize the automata
        for(int i = 1; i < numberLayers; ++i) {
            ta_layer_list.add(initializeAtom());
        }

        clauseWeights = initializeClauseWeights();

        automata = new GraphAutomataUpdates(this);
    }

    private int[][] initializeClauseWeights() {
        int[][] clauseWeights = new int[nClasses][nClauses];

        for (int j = 0; j < nClasses; ++j) {
            for (int k = 0; k < nClauses; ++k) {
                //randomly initialize the clause with -1 or 1
                if (rng.nextDouble() < 0.5) {
                    clauseWeights[j][k] = -1;
                } else {
                    clauseWeights[j][k] = 1;
                }
            }
        }

        return clauseWeights;
    }

    private int[][] initializeAtom() {

        int[][] ta_atom_state = new int[nClauses][la_chunks*state_bits];

        System.out.println("Initializing atom state with " + nClauses + " clauses and " + la_chunks + " chunks of state bits: " + state_bits);

        for (int j = 0; j < nClauses; ++j) {
            for (int k = 0; k < la_chunks; ++k) {
                for (int b = 0; b < state_bits-1; ++b) {
                    ta_atom_state[j][k*state_bits + b] = ~0;
                }
                ta_atom_state[j][k*state_bits + state_bits-1] = 0;
            }
        }

        return ta_atom_state;
    }


    /**
     * Evaluate the vertex layer for a given input X of vertex data from the graph
     * @param X
     * @param globalClauseNodeOutput
     */
    public int[] evaluateVertexLayer(int[] X, int[][] globalClauseNodeOutput) {

        return automata.calculateMessages(ta_atom_state,
                null,
                0,
                globalClauseNodeOutput,
                X);

    }

    /**
     * Evaluate a layer of the graph given the input X (layer data) and conditional on the previous layer clause node output
     * @param X
     * @param globalClauseNodeInput
     * @param globalClauseNodeOutput
     * @param numberOfIncludeActions
     * @param layer 1,2,...,numberLayers-1
     */
    public void evaluateLayer(int[] X, int[][] globalClauseNodeInput, int[][] globalClauseNodeOutput, int[] numberOfIncludeActions, int layer) {

        //check that the layer is valid
        if(layer < 0 || layer >= numberLayers) {
            throw new IllegalArgumentException("Layer " + layer + " is out of bounds. Must be between 0 and " + (numberLayers-1));
        }

        automata.calculateMessagesConditional(ta_layer_list.get(layer-1),
                null,
                0,
                number_of_vertices,
                globalClauseNodeInput,
                globalClauseNodeOutput,
                numberOfIncludeActions,
                X);
    }

    /**
     * Given the global clause node output, evaluate the automata using the clause weights
     * @param globalClauseNodeOutput
     * @return int[] of size nClasses
     */
    public int[] evaluate(int[][] globalClauseNodeOutput) {
        return automata.evaluate(globalClauseNodeOutput, clauseWeights);
    }

    /**
     * Count the number of literals in the global clause node output
     * @param globalClauseNodeOutput
     * @return int[] is a vector of all the literals counted in the globalClauseNodeOutput
     */
    public int[][] predictLiteralCount(int[][] globalClauseNodeOutput, int[] classSum) {
        return automata.countIncludedLiterals(globalClauseNodeOutput, ta_atom_state,  clauseWeights, classSum);
    }


    public void selectClauseNode(int[][] globalClauseNodeOutput) {
        int[] clauseNode = automata.selectClauseNode(globalClauseNodeOutput);
    }

    public void selectClauseUpdate(int[][] X, int[] clauseNode, int[] numberOfIncludeActions, int[] classSum, int[] y) {

        int[][] classClauseUpdate = automata.selectClauseUpdates(clauseWeights, classSum, y, clauseNode);

        automata.update(s,
                ta_atom_state,        // dims: [CLAUSES][LA_CHUNKS]
                clauseNode,               // length == CLAUSES
                numberOfIncludeActions,   // length == CLAUSES
                X[0],                      // dims: [numberOfGraphs][LA_CHUNKS]
                classClauseUpdate);       // dims: [CLASSES][CLAUSES]


        //update next layers
        for(int i = 0; i < numberLayers-1; ++i) {
            automata.updateMessage(s,
                    ta_layer_list.get(i),        // dims: [CLAUSES][LA_CHUNKS]
                    clauseNode,               // length == CLAUSES
                    numberOfIncludeActions,   // length == CLAUSES
                    X[i+1],                      // dims: [numberOfGraphs][LA_CHUNKS]
                    classClauseUpdate);       // dims: [CLASSES][CLAUSES]

        }
    }

    /**
     * Update the automata with the given input X, y, classSum, globalClauseNodeOutput and numberOfIncludeActions
     * @param X
     * @param y
     * @param classSum
     * @param globalClauseNodeOutput
     * @param numberOfIncludeActions
     */
    public void update(int[][] X, int[] y, int[] classSum, int[][] globalClauseNodeOutput, int[] numberOfIncludeActions) {

        //select the clause node
        int[] clauseNode = automata.selectClauseNode(globalClauseNodeOutput);

        selectClauseUpdate(X, clauseNode, numberOfIncludeActions, classSum, y);

    }



    public void fitOneLyer(int[][][] X, int[] y) {

        //create one-hot encoding for y
        int[][] y_one_hot = new int[y.length][nClasses];
        for(int i = 0; i < y.length; i++) {
            y_one_hot[i][y[i]] = 1;
        }

        int[][] globalClauseNodeOutput = new int[nClauses][nodeChunks];


        int correct = 0;
        int total = 0;
        for(int k = 0; k < X.length; k++) {

            int index = k;

            //System.out.println("Index: " + index + " X: " + X[index][0].length + " y: " + y[index]);


            //first need to evaluate to get the output clause vertex map and the class_sum output
            int[] numberOfIncludeActions = evaluateVertexLayer(X[index][0], globalClauseNodeOutput);

            int[] class_sums = evaluate(globalClauseNodeOutput);

            update(X[index], y_one_hot[index], class_sums, globalClauseNodeOutput, numberOfIncludeActions);
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
            total += 1;
        }
        //compute the accuracy
        System.out.println("Accuracy: " + ((float) correct / (float)total));
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
            int[] numberOfIncludeActions = evaluateVertexLayer(X[index][0], globalClauseNodeOutput);
            int[] class_sums = evaluate(globalClauseNodeOutput);

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





    public void setMaxNumberOfLiterals(int max_literals) {
        this.max_literals = max_literals;
    }

    public boolean isFocused_negative_sampling() {
        return focused_negative_sampling;
    }

    public void setFocused_negative_sampling(boolean focused_negative_sampling) {
        this.focused_negative_sampling = focused_negative_sampling;
    }

    public int[][] getTa_atom_state() {
        return ta_atom_state;
    }




    public boolean isBoost() {
        return boost;
    }

    public int getLa_chunks() {
        return la_chunks;
    }

    public int getMax_literals() {
        return max_literals;
    }

    public int getnClauses() {
        return nClauses;
    }

    public int getnClasses() {
        return nClasses;
    }

    public int getThreshold() {
        return threshold;
    }


    public int getFilter() {
        return this.filter;
    }

    public int getNumberNodes() {
        return this.number_of_vertices;
    }

    public int getFeature_bits() {
        return this.number_of_features;
    }

    public float getS() {
        return s;
    }

    public int getNumberLayers() {
        return numberLayers;
    }
}
