package encoding.embedding;


import encoding.hypervector.SparseBinaryVector;

import java.io.Serializable;
import java.util.*;
import java.util.stream.IntStream;

import static util.HV.RNG;


/**
 * FixedLevelEmbedding is an implementation of the Embedding interface
 * that provides a fixed number of levels of hypervectors (for example, for different price levels in an orderbook)
 */

public class FixedLevelEmbedding implements Embedding<Integer>, Serializable {
    private static final long serialVersionUID = 1L;
    private final List<SparseBinaryVector> levels;

    public FixedLevelEmbedding(int levelsCount) {
        this.levels = new ArrayList<>();
        SparseBinaryVector base = SparseBinaryVector.randVector();
        this.levels.add(base);

        for (int i = 1; i < levelsCount; i++) {
            boolean[] bits = levels.get(0).toBooleanVector().clone();
            Set<Integer> flipped = new HashSet<>();
            while (flipped.size() < (SparseBinaryVector.DIMENSION / levelsCount) * i) {
                flipped.add(RNG.nextInt(SparseBinaryVector.DIMENSION));
            }
            for (int idx : flipped) {
                bits[idx] = !bits[idx];
            }
            this.levels.add(SparseBinaryVector.fromBooleanVector(bits));
        }
    }


    @Override
    public SparseBinaryVector forward(Integer level) {
        //level should be between 0 and levels.size()-1, otherwise, send the last level in the list if greater
        if (level < 0) {
            level = 0;
        } else if (level >= levels.size()) {
            level = levels.size() - 1;
        }
        return levels.get(level);
    }

    @Override
    public SparseBinaryVector forward(int level) {
        //level should be between 0 and levels.size()-1, otherwise, send the last level in the list if greater
        if (level < 0) {
            level = 0;
        } else if (level >= levels.size()) {
            level = levels.size() - 1;
        }
        return levels.get(level);
    }


    @Override
    public Integer back(SparseBinaryVector hv) {
        return IntStream.range(0, levels.size())
                .boxed()
                .min(Comparator.comparingInt(i -> hv.hammingDistance(levels.get(i))))
                .orElse(0);
    }
}
