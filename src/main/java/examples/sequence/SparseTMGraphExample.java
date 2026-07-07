package examples.sequence;

import encoding.embedding.IntervalEmbedding;
import encoding.hypervector.SparseBinaryVector;
import tsetlin.GraphEncoder;
import tsetlin.TsetlinAttentionLearning;
import util.BitUtils;
import util.HV;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


/**
 * A simple example to demonstrate the use of Tsetlin graph
 * with sparse binary encoding to learn a simple function
 * of three random numbers.
 *
 * Demonstrates the proof of concept of the encoding using both binding and bundling
 */
public class SparseTMGraphExample {


    int nClauses = 10;
    int threshold = 10;
    float s = 1f;
    int maxLiterals = 5;
    Random random = new Random();

    int[][][] X_train;
    int[] y_train;
    int[][][] X_test;
    int[] y_test;



    private final IntervalEmbedding intervalEmbedding = new IntervalEmbedding(0, 20,20);
    private final GraphEncoder encoder = new GraphEncoder(HV.DIMENSION);

//    private final SparseBinaryVector edge1 = SparseBinaryVector.withFixedSegments(0, 4);
//    private final SparseBinaryVector edge2 = SparseBinaryVector.withFixedSegments(5, 9);
//    private final SparseBinaryVector edge3 = SparseBinaryVector.withFixedSegments(10, 14);

    public void createData(int samples) {


        final SparseBinaryVector edge1 = SparseBinaryVector.withFixedSegments(0, 4);
        final SparseBinaryVector edge2 = SparseBinaryVector.withFixedSegments(5, 9);
        final SparseBinaryVector edge3 = SparseBinaryVector.withFixedSegments(10, 14);


        //randomly sample 0 - 20;
        int[][][] X = new int[samples][1][];
        int[] y = new int[samples];

        for(int i = 0; i < samples; i++) {


            int[] randomValueArray = new int[3];
            randomValueArray[0] = random.nextInt(20);
            randomValueArray[1] = random.nextInt(20);
            randomValueArray[2] = random.nextInt(20);

            SparseBinaryVector projected1 = intervalEmbedding.forward(randomValueArray[0]).bind(edge1);
            SparseBinaryVector projected2 = intervalEmbedding.forward(randomValueArray[1]).bind(edge2);
            SparseBinaryVector projected3 = intervalEmbedding.forward(randomValueArray[2]).bind(edge3);

            List<SparseBinaryVector> projected = new ArrayList<>();
            projected.add(projected1);
            projected.add(projected2);
            projected.add(projected3);

            //logic majority
            SparseBinaryVector myProjected = SparseBinaryVector.bundle(projected);

            X[i][0] = BitUtils.longsToInts(myProjected.encoded());

            //if sum of randomValueArray is less than 30, then class 0 else class 1
//            if(randomValueArray[0] + randomValueArray[1] + randomValueArray[2] < 30) {
//                y[i] = 0;
//            } else {
//                y[i] = 1;
//            }

            //if sum of randomValueArray is less than 20, if between 20 and 40 class 1, else class 2
            if(randomValueArray[0] + randomValueArray[1] + randomValueArray[2] < 20) {
                y[i] = 0;
            } else if(randomValueArray[0] + randomValueArray[1] + randomValueArray[2] < 40) {
                y[i] = 1;
            } else {
                y[i] = 2;
            }

        }



        //take 80 percent of the data for training
        int trainSize = (int)(X.length * .8);
        //build a random set of unique indices from 1 - X.length
        int[] indices = new int[X.length];
        for(int i = 0; i < X.length; i++) {
            indices[i] = i;
        }
        //shuffle the indices
        for(int i = 0; i < X.length; i++) {
            int j = (int)(random.nextDouble() * X.length);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        X_train = new int[trainSize][1][];
        y_train = new int[trainSize];
        for(int i = 0; i < trainSize; i++) {
            X_train[i][0] = X[indices[i]][0];
            y_train[i] = y[indices[i]];
        }

        X_test = new int[X.length - trainSize][1][];
        y_test = new int[X.length - trainSize];

        for(int i = trainSize; i < X.length; i++) {
            X_test[i - trainSize][0] = X[indices[i]][0];
            y_test[i - trainSize] = y[indices[i]];
        }
    }



    public void createDataSimple(int samples) {

        //randomly sample 0 - 20;
        int[][][] X = new int[samples][1][];
        int[] y = new int[samples];

        for(int i = 0; i < samples; i++) {


            int[] randomValueArray = new int[3];
            randomValueArray[0] = random.nextInt(20);
            randomValueArray[1] = random.nextInt(20);
            randomValueArray[2] = random.nextInt(20);

//            SparseBinaryVector projected1 = intervalEmbedding.forward(randomValueArray[0]).bind(edge1);
//            SparseBinaryVector projected2 = intervalEmbedding.forward(randomValueArray[1]).bind(edge2);
//            SparseBinaryVector projected3 = intervalEmbedding.forward(randomValueArray[2]).bind(edge3);

            SparseBinaryVector projected1 = intervalEmbedding.forward(randomValueArray[0]);
//            SparseBinaryVector projected2 = intervalEmbedding.forward(randomValueArray[1]);
//            SparseBinaryVector projected3 = intervalEmbedding.forward(randomValueArray[2]);

            List<SparseBinaryVector> projected = new ArrayList<>();
            projected.add(projected1);
//            projected.add(projected2);
//            projected.add(projected3);

            //logic majority
            SparseBinaryVector myProjected = SparseBinaryVector.bundle(projected);

            X[i][0] = BitUtils.longsToInts(myProjected.encoded());

            //if sum of randomValueArray is less than 30, then class 0 else class 1

//            if(randomValueArray[0] + randomValueArray[1] + randomValueArray[2] < 30) {
//                y[i] = 0;
//            } else {
//                y[i] = 1;
//            }

            y[i] = randomValueArray[0] > 10 ? 1 : 0; // simple binary classification based on first value

        }

        //take 80 percent of the data for training
        int trainSize = (int)(X.length * .8);
        //build a random set of unique indices from 1 - X.length
        int[] indices = new int[X.length];
        for(int i = 0; i < X.length; i++) {
            indices[i] = i;
        }
        //shuffle the indices
        for(int i = 0; i < X.length; i++) {
            int j = (int)(random.nextDouble() * X.length);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        X_train = new int[trainSize][1][];
        y_train = new int[trainSize];
        for(int i = 0; i < trainSize; i++) {
            X_train[i][0] = X[indices[i]][0];
            y_train[i] = y[indices[i]];
        }

        X_test = new int[X.length - trainSize][1][];
        y_test = new int[X.length - trainSize];

        for(int i = trainSize; i < X.length; i++) {
            X_test[i - trainSize][0] = X[indices[i]][0];
            y_test[i - trainSize] = y[indices[i]];
        }
    }

    public void train() {

        int nClasses = 3;
        int nClauses = 60;
        int threshold = 100;
        float max_specificity = 5.0f;
        boolean boost = true;
        int max_literals = 80;


        TsetlinAttentionLearning model = new TsetlinAttentionLearning(encoder,  nClauses,  nClasses, threshold,  max_specificity,  boost, max_literals);


        for(int e = 0; e < 6; e++) {

            model.fitOneLyer(X_train, y_train);

            model.predict(X_test, y_test);
        }

    }

    //main
    public static void main(String[] args) {

        SparseTMGraphExample encoding = new SparseTMGraphExample();
        encoding.createData(1000);
        encoding.train();
    }

}