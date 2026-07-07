package tsetlin;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import static graph.TsetlinAttentionGraph.and;

public class GraphAttentionLearning {

    //logger
    private static Logger logger = LoggerFactory.getLogger(GraphAttentionLearning.class);

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

    private int max_literals = 0;


    private int numberLayers = 1; //number of layers in the graph

    private int number_of_vertices; //total vertices used for the graph
    private int number_of_ta_chunks;
    private int number_of_features;
    private int dim_x;
    private Random rng = new Random(21);

    private GraphAutomata[] tm;
    private int[][][] ta_atom_state; //[CLAUSES][LA_CHUNKS][STATE_BITS];
    private int[][][] message_layer;



    private int filter;


    private boolean focused_negative_sampling = false;




    public GraphAttentionLearning(GraphEncoder graphEncoder, int nClauses, int nClasses, int threshold, float max_specificity, boolean boost, float clause_drop_p) {

        this.nClauses = nClauses;
        this.nClasses = nClasses;
        this.threshold = threshold;
        this.max_specificity = max_specificity;
        this.boost = boost;
        this.drop_clause_p = clause_drop_p;

        this.number_of_vertices = graphEncoder.getNumber_of_vertices();
        this.dim_x = graphEncoder.getDim_x();
        this.number_of_features = graphEncoder.getTotalNumberFeatures();
        this.number_of_ta_chunks = graphEncoder.getLayer_ta_chunks();
        this.la_chunks = graphEncoder.getLayer_ta_chunks();
        this.numberLayers = graphEncoder.getNumberLayers();

        //print out all the parameters
//        System.out.println("nClauses: " + nClauses);
//        System.out.println("nClasses: " + nClasses);
//        System.out.println("threshold: " + threshold);
//        System.out.println("max_specificity: " + max_specificity);
//        System.out.println("boost: " + boost);
//        System.out.println("drop_clause_p: " + drop_clause_p);
//        System.out.println("number_of_vertices: " + number_of_vertices);
//        System.out.println("number_of_ta_chunks: " + number_of_ta_chunks);
//        System.out.println("number_of_features: " + number_of_features);
//        System.out.println("dim_x: " + dim_x);
//        System.out.println("la_chunks: " + la_chunks);
//        System.out.println("state_bits: " + state_bits);

        logger.info("nClauses: {}", nClauses);
        logger.info("nClasses: {}", nClasses);
        logger.info("threshold: {}", threshold);
        logger.info("max_specificity: {}", max_specificity);
        logger.info("number_of_vertices: {}", number_of_vertices);
        logger.info("number_of_ta_chunks: {}", number_of_ta_chunks);
        logger.info("number_of_features: {}", number_of_features);
        logger.info("dim_x: {}", dim_x);
        logger.info("la_chunks: {}", la_chunks);

        initializeAtom();

        tm = new GraphAutomata[nClasses];

        for(int i = 0; i < nClasses; i++) {

            tm[i] = new GraphAutomata(graphEncoder, threshold, nClauses, max_specificity, boost, drop_clause_p)
                    .initializeVertexLayer(ta_atom_state);

            for(int layer = 0; layer < numberLayers-1; layer++) {
                tm[i].initializeMessageLayer(initializeMessageLayer());
            }
        }

        if (((graphEncoder.getFeatureDimension()*2) % INT_SIZE) != 0) {
            this.filter  = (~(0xffffffff << ((graphEncoder.getFeatureDimension()*2) % INT_SIZE)));
        } else {
            this.filter = 0xffffffff;
        }


    }

    private void initializeAtom() {

        ta_atom_state = new int[nClauses][la_chunks][state_bits];

        for (int j = 0; j < nClauses; ++j) {
            for (int k = 0; k < la_chunks; ++k) {
                for (int b = 0; b < state_bits-1; ++b) {
                    ta_atom_state[j][k][b] = ~0;
                }
                ta_atom_state[j][k][state_bits-1] = 0;
            }
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

                    if(p == y[batch_example]) {
                        correct++;
                    }
                    count++;
                }
                //System.out.println("accuracy " + (float)correct/count);
            }

        }
    }

    public void fitGraph(int[][][] X, int[] y, boolean output_balancing) {

        int[] class_observed = new int[nClasses];
        int[] example_indexes = new int[nClasses];
        int example_counter = 0;

        ArrayList<Integer> example_indexes_list = new ArrayList<Integer>();
        //a shuffled list of indices for the examples X.length
        for(int i = 0; i < X.length; i++) {
            example_indexes_list.add(i);
        }
        Collections.shuffle(example_indexes_list);

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
                    int p = updateGraph(X[batch_example], y[batch_example]);

                    //System.out.println("Example: " + batch_example + " Class: " + y[batch_example] + " Prediction: " + p);
                }
            }
        }
    }




    public int update(int[] Xi, int target_class) {

        //throw error if nClasses == 1
        if(nClasses == 1) {
            throw new IllegalArgumentException("nClasses == 1, use regression");
        }

        tm[target_class].update(Xi, 1);

        int negative_target_class = rng.nextInt(nClasses);

        if(focused_negative_sampling) {

            //for each tm sum the sumOfWeights
            float[] sumOfWeights = new float[nClasses];
            float sum = 0;
            for (int i = 0; i < nClasses; i++) {
                sumOfWeights[i] = tm[i].getSumOfWeights();
                sum += sumOfWeights[i];
            }
            //find the sum of sumOfWeights

            for (int i = 0; i < nClasses; i++) {
                sumOfWeights[i] = sumOfWeights[i] / sum;
            }


            //generate a random integer between 0 and nClasses with probability sumOfWeights
            for (int i = 1; i < nClasses; i++) {
                if (rng.nextFloat() < sumOfWeights[i]) {
                    negative_target_class = i;
                    break;
                }
            }
        }

        while(negative_target_class == target_class) {
            negative_target_class = rng.nextInt(nClasses);
        }
        tm[negative_target_class].update(Xi, 0);

//		//print out weights
//		int[] w = tm[target_class].getClause_weights();
//		for (int j = 0; j < w.length; j++) {
//			System.out.print(w[j] + " ");
//		}
//		System.out.println("\n target " + target_class);
//
//		int pred = predict(Xi);
//		System.out.println("\n predict " + pred + " " + target_class + " " + negative_target_class);


        return predict(Xi);
    }


    public int updateGraph(int[][] Xi, int[] class_output, int target_class) {

        tm[target_class].updateGraph(Xi,  1, class_output[target_class]);
        int negative_target_class = rng.nextInt(nClasses);
        while(negative_target_class == target_class) {
            negative_target_class = rng.nextInt(nClasses);
        }
        tm[negative_target_class].updateGraph(Xi, 0, class_output[negative_target_class]);

        int[][] vertex_clause_map = evaluateVertexClauseMap(Xi[0], ta_atom_state);
        int[][] vertex_clause_map_2 = evaluateVertexClauseMap(Xi[1], message_layer);

        int[][] vertex_clause = and(vertex_clause_map, vertex_clause_map_2);

        int pred = evaluate(vertex_clause);
        return pred;

    }

    public int updateGraph(int[][] Xi, int target_class) {

        //throw error if nClasses == 1
        if(nClasses == 1) {
            throw new IllegalArgumentException("nClasses == 1, use regression");
        }

        tm[target_class].updateGraph(Xi, 1);

        int negative_target_class = rng.nextInt(nClasses);

        if(focused_negative_sampling) {

            //for each tm sum the sumOfWeights
            float[] sumOfWeights = new float[nClasses];
            float sum = 0;
            for (int i = 0; i < nClasses; i++) {
                sumOfWeights[i] = tm[i].getSumOfWeights();
                sum += sumOfWeights[i];
            }
            //find the sum of sumOfWeights

            for (int i = 0; i < nClasses; i++) {
                sumOfWeights[i] = sumOfWeights[i] / sum;
            }


            //generate a random integer between 0 and nClasses with probability sumOfWeights
            for (int i = 1; i < nClasses; i++) {
                if (rng.nextFloat() < sumOfWeights[i]) {
                    negative_target_class = i;
                    break;
                }
            }
        }

        while(negative_target_class == target_class) {
            negative_target_class = rng.nextInt(nClasses);
        }
        tm[negative_target_class].updateGraph(Xi, 0);

        return predictGraph(Xi);
    }


    public int updateGraph(int[][] Xi, int[][] clause_vertex_map, int target_class) {



        return 0;

    }



    public int predict(int[] X) {

        //throw error if nClasses == 1
        if(nClasses == 1) {
            throw new IllegalArgumentException("nClasses == 1, use regression");
        }

        int max_class_sum = tm[0].score(X);

        int max_class = 0;
        for (int i = 1; i < nClasses; i++) {
            int class_sum = tm[i].score(X);
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

    public int evaluate(int[][] vertex_clause_map) {

        int max_class_sum = tm[0].getClassSumGraph(vertex_clause_map);

        int max_class = 0;
        for (int i = 1; i < nClasses; i++) {
            int class_sum = tm[i].getClassSumGraph(vertex_clause_map);
            if (max_class_sum < class_sum) {
                max_class_sum = class_sum;
                max_class = i;
            }
        }
        return max_class;
    }

    public int[] evaluateSums(int[][] vertex_clause_map) {

        int[] class_sum = new int[nClasses];
        for (int i = 0; i < nClasses; i++) {
            class_sum[i] = tm[i].getClassSumGraph(vertex_clause_map);
        }
        return class_sum;
    }


    public int predictGraph(int[][] X) {

        //throw error if nClasses == 1
        if(nClasses == 1) {
            throw new IllegalArgumentException("nClasses == 1, use regression");
        }

        int max_class_sum = tm[0].scoreGraph(X);

        int max_class = 0;
        for (int i = 1; i < nClasses; i++) {
            int class_sum = tm[i].scoreGraph(X);
            if (max_class_sum < class_sum) {
                max_class_sum = class_sum;
                max_class = i;
            }
        }
        return max_class;
    }





    public void setMaxNumberOfLiterals(int max_literals) {
        this.max_literals = max_literals;
        for (int i = 0; i < nClasses; i++) {
            tm[i].setMax_number_literals(max_literals);
        }
    }

    public boolean isFocused_negative_sampling() {
        return focused_negative_sampling;
    }

    public void setFocused_negative_sampling(boolean focused_negative_sampling) {
        this.focused_negative_sampling = focused_negative_sampling;
    }

    public int[][][] getTa_atom_state() {
        return ta_atom_state;
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
