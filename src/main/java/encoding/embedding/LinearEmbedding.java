package encoding.embedding;


import encoding.hypervector.SparseBinaryVector;

import java.io.Serializable;
import java.util.*;
import java.util.stream.IntStream;

import static util.HV.RNG;


/**
 * LinearEmbedding is an implementation of the Embedding interface
 * that provides a mapping from a continuous range of values
 * to a set of hypervectors representing intervals.
 *
 * This embedding is useful for representing continuous data and is distance preserving,
 * For example, it can be used to represent volume levels in an order book at a given price
 */
public class LinearEmbedding implements Embedding<Double>,  Serializable {

    private static final long serialVersionUID = 1L;
    private final List<SparseBinaryVector> intervals;
    private final double low;
    private final double high;
    private final double step;
    private final double SLACK = 0.001;

    private static double CONTINUITY_FLIP_RATE = 0.02;  // 2% of bits for gradual change
    private static final double NOISE_FLIP_RATE = 0.07;      // 0.5% of bits for separation


    public LinearEmbedding(double low, double high, int divisions) {
        this.low = low - SLACK;
        this.high = high + SLACK;
        this.step = (this.high - this.low) / divisions;
        this.intervals = new ArrayList<>();

        SparseBinaryVector base = SparseBinaryVector.randVector();
        this.intervals.add(base);

        for (int i = 1; i < divisions; i++) {
            boolean[] previous = intervals.get(i - 1).toBooleanVector();
            boolean[] next = Arrays.copyOf(previous, previous.length);

            int totalBits = SparseBinaryVector.DIMENSION;

            CONTINUITY_FLIP_RATE = 1.0/divisions;

            // Flip bits for smooth change
            int numToFlip = (int) (totalBits * CONTINUITY_FLIP_RATE);
            Set<Integer> flipped = new HashSet<>();
            while (flipped.size() < numToFlip) {
                int idx = RNG.nextInt(totalBits);
                next[idx] = !next[idx];
                flipped.add(idx);
            }

            // Add noise (flip a few *new* random bits)
            int numNoise = (int) (totalBits * NOISE_FLIP_RATE);
            while (flipped.size() < numToFlip + numNoise) {
                int idx = RNG.nextInt(totalBits);
                if (!flipped.contains(idx)) {
                    next[idx] = !next[idx];
                    flipped.add(idx);
                }
            }

            this.intervals.add(SparseBinaryVector.fromBooleanVector(next));
        }
    }


    @Override
    public SparseBinaryVector forward(Double x) {
        if (x < low) x = low + SLACK;
        else if (x > high) x = high - SLACK;

        int index = (int) ((x - low) / step);
        return intervals.get(index);
    }

    @Override
    public SparseBinaryVector forward(int level) {
        double x = (double)level;
        if (x < low) x = low + SLACK;
        else if (x > high) x = high - SLACK;

        int index = (int) ((x - low) / step);
        return intervals.get(index);
    }

    public Double back(SparseBinaryVector hv) {
        int index = IntStream.range(0, intervals.size())
                .boxed()
                .min(Comparator.comparingInt(i -> hv.hammingDistance(intervals.get(i))))
                .orElse(0);
        return low + index * step;
    }
}
