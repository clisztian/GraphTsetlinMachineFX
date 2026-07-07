package tsetlin;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;


public class FuzzyAttentionLearning {

    //logger
    private static Logger logger = LoggerFactory.getLogger(FuzzyAttentionLearning.class);

    final int INT_SIZE = 32;
    private final int STATE_BITS = 8;

    private final int nClauses;



    private final int nClasses;
    private final int threshold;
    private final boolean boost;
    private final int state_bits = 8;
    private final int la_chunks;

    private int max_literals = 0;


    private int numberLayers = 1; //number of layers in the graph

    private int number_of_vertices; //total vertices used for the graph
    private int number_of_ta_chunks;
    private int number_of_features;
    private int dim_x;
    private Random rng = new Random(21);

    private TMClassifier[] tm;
    private int[][][] message_layer;



    private int filter;


    private boolean focused_negative_sampling = false;




    public FuzzyAttentionLearning(GraphEncoder graphEncoder, int nClauses, int nClasses, int threshold, float max_specificity, boolean boost, int LF, int max_literals) {

        this.nClauses = nClauses;
        this.nClasses = nClasses;
        this.threshold = threshold;
        this.boost = boost;
        this.number_of_vertices = graphEncoder.getNumber_of_vertices();
        this.dim_x = graphEncoder.getDim_x();
        this.number_of_features = graphEncoder.getTotalNumberFeatures();
        this.number_of_ta_chunks = graphEncoder.getLayer_ta_chunks();
        this.la_chunks = graphEncoder.getLayer_ta_chunks();
        this.numberLayers = graphEncoder.getNumberLayers();


        logger.info("nClauses: {}", nClauses);
        logger.info("nClasses: {}", nClasses);
        logger.info("threshold: {}", threshold);
        logger.info("max_specificity: {}", max_specificity);
        logger.info("number_of_vertices: {}", number_of_vertices);
        logger.info("number_of_ta_chunks: {}", number_of_ta_chunks);
        logger.info("number_of_features: {}", number_of_features);
        logger.info("dim_x: {}", dim_x);
        logger.info("la_chunks: {}", la_chunks);



        tm = new TMClassifier[nClasses];

        for(int i = 0; i < nClasses; i++) {
            tm[i] = new TMClassifier(threshold, graphEncoder.getFeatureDimension(), nClauses, max_specificity,  boost,   LF,  max_literals).initialize();
        }

        if (((graphEncoder.getFeatureDimension()*2) % INT_SIZE) != 0) {
            this.filter  = (~(0xffffffff << ((graphEncoder.getFeatureDimension()*2) % INT_SIZE)));
        } else {
            this.filter = 0xffffffff;
        }


    }


    private int[][][] initializeMessageLayer() {

        message_layer = new int[nClauses][la_chunks][state_bits];

        for (int j = 0; j < nClauses; ++j) {
            for (int k = 0; k < la_chunks; ++k) {
                for (int b = 0; b < state_bits-1; ++b) {
                    message_layer[j][k][b] = ~0;
                }
                message_layer[j][k][state_bits-1] = 0;
            }
        }
        return message_layer;
    }


    /**
     * Fit one epoch on the given data
     * @param X
     * @param y
     * @param output_balancing
     */
    public void fit(int[][] X, int[] y, boolean output_balancing) {

        int[] class_observed = new int[nClasses];
        int[] example_indexes = new int[nClasses];
        int example_counter = 0;

        ArrayList<Integer> example_indexes_list = new ArrayList<Integer>();
        //a shuffled list of indices for the examples X.length
        for(int i = 0; i < X.length; i++) {
            example_indexes_list.add(i);
        }
        Collections.shuffle(example_indexes_list);

        int correct = 0;
        int count = 0;
        for(int i = 0; i < X.length; i++) {

            int e = example_indexes_list.get(i);

            if(output_balancing) {

                if(class_observed[y[e]] == 0) {
                    example_indexes[y[e]] = e;
                    class_observed[y[e]] = 1;
                    example_counter++;
                }
            }
            else {
                example_indexes[example_counter] = e;
                example_counter++;
            }

            if(example_counter == nClasses) {

                example_counter = 0;

                for(int j = 0; j < nClasses; j++) {
                    class_observed[j] = 0;
                    int batch_example = example_indexes[j];
                    int p = update(X[batch_example], y[batch_example]);
                    //System.out.println(i + " Class: " + y[batch_example] + " Prediction: " + p);

                    //log the prediction and the actual class
                    //logger.info("Class: {}, Prediction: {}", y[batch_example], p);

                    if(p == y[batch_example]) {
                        correct++;
                    }
                    count++;
                }
                //System.out.println("accuracy " + (float)correct/count);
            }

        }
    }





    public int update(int[] Xi, int target_class) {

        //throw error if nClasses == 1
        if(nClasses == 1) {
            throw new IllegalArgumentException("nClasses == 1, use regression");
        }

        tm[target_class].updateFuzzy(Xi, 1);

        int negative_target_class = rng.nextInt(nClasses);


        while(negative_target_class == target_class) {
            negative_target_class = rng.nextInt(nClasses);
        }
        tm[negative_target_class].updateFuzzy(Xi, 0);



        return predict(Xi);
    }




    public int predict(int[] X) {

        //throw error if nClasses == 1
        if(nClasses == 1) {
            throw new IllegalArgumentException("nClasses == 1, use regression");
        }

        int max_class_sum = tm[0].scoreFuzzy(X);

        //System.out.println("Class 0: " + max_class_sum);

        int max_class = 0;
        for (int i = 1; i < nClasses; i++) {
            int class_sum = tm[i].scoreFuzzy(X);

            //System.out.println("Class " + i + ": " + class_sum);

            if (max_class_sum < class_sum) {
                max_class_sum = class_sum;
                max_class = i;
            }
        }
        return max_class;
    }


    public int[][] evaluateVertexClauseMap(int[] X, int[][][] ta_atom_state) {

        //initialize the list of lists for each vertex
        int[][] vertex_clause_map = new int[number_of_vertices][nClauses];

        for (int j = 0; j < nClauses; j++) {

            for(int vertex = 0; vertex < number_of_vertices; ++vertex) {

                int output = 1;
                for (int k = 0; k < la_chunks-1; k++) {
                    output = (output == 1) && ((ta_atom_state[j][k][STATE_BITS -1] & X[vertex*la_chunks + k]) == ta_atom_state[j][k][STATE_BITS -1]) ? 1 : 0;

                    if (output == 0) {
                        break;
                    }
                }

                output = (output == 1) && ((ta_atom_state[j][la_chunks-1][STATE_BITS -1] & X[vertex*la_chunks + la_chunks-1] & filter) ==
                        (ta_atom_state[j][la_chunks-1][STATE_BITS -1] & filter)) ? 1 : 0;

                if (output == 1) {
                    vertex_clause_map[vertex][j] = 1;
                }
            }
        }
        return vertex_clause_map;
    }






    public boolean isFocused_negative_sampling() {
        return focused_negative_sampling;
    }

    public void setFocused_negative_sampling(boolean focused_negative_sampling) {
        this.focused_negative_sampling = focused_negative_sampling;
    }


    public int[][][] getMessage_layer() {
        return message_layer;
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

    public int getFeature_bits() {
        return dim_x;
    }

    public int getFilter() {
        return this.filter;
    }

    public int getNumberNodes() {
        return this.number_of_vertices;
    }
}
