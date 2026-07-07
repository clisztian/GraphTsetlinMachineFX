package util;

import java.util.Random;

/**
 * Hyperdimensional Vector (HV) utility class for generating random vectors and calculating Hamming distance.
 * This class provides methods to create random hyperdimensional vectors and compute the Hamming distance between them.
 */
public class HV {

    public static final int SEED = 34; // Seed for reproducibility
    public static final float SPARSITY = 0.1f; // Example sparsity, adjust as needed
    public static final int SEGMENT_SIZE = 25; // Example segment size, adjust as needed
    public static final int LONG_SIZE = Long.SIZE; // 64
    public static final int DIMENSION = SEGMENT_SIZE*LONG_SIZE; // Example dimension, adjust as needed
    public static final int K = (int) (DIMENSION * SPARSITY); // Example sparsity, adjust as needed
    public static final int SIZE = 8; // a factor of DIMENSION
    public static final Random RNG = new Random(SEED); // Random number generator with a seed

    public static final int ARRAY_SIZE = (int) Math.ceil((double) DIMENSION / LONG_SIZE);

    public static long[] generateRandomVector() {
        long[] vector = new long[ARRAY_SIZE];
        for (int i = 0; i < ARRAY_SIZE; i++) {
            vector[i] = (long) (Math.random() * Long.MAX_VALUE);
        }
        return vector;
    }

    public static int hammingDistance(long[] a, long[] b) {
        int distance = 0;
        for (int i = 0; i < ARRAY_SIZE; i++) {
            distance += Long.bitCount(a[i] ^ b[i]);
        }
        return distance;
    }

}
