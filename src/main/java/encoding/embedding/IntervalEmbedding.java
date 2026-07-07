package encoding.embedding;


import encoding.hypervector.SparseBinaryVector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

/**
 * LinearEmbedding is an implementation of the Embedding interface
 * that provides a mapping from a continuous range of values
 * to a set of hypervectors representing intervals.
 *
 * This embedding is useful for representing continuous data and is distance preserving,
 * For example, it can be used to represent volume levels in an order book at a given price
 */
public class IntervalEmbedding implements Embedding<Double>,  Serializable {

    private static final long serialVersionUID = 1L;
    private final List<SparseBinaryVector> intervals;
    private final double low;
    private final double high;
    private final double step;
    private final double SLACK = 0.001;

    public IntervalEmbedding(double low, double high, int divisions) {

        this.low = low - SLACK;
        this.high = high + SLACK;
        this.step = (this.high - this.low) / divisions;
        this.intervals = new ArrayList<>();

        for (int i = 0; i < divisions; i++) {
            this.intervals.add(SparseBinaryVector.randVector());
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
