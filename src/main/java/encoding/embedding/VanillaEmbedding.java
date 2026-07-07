package encoding.embedding;



import encoding.hypervector.SparseBinaryVector;

import java.io.Serializable;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;


/**
 * VanillaEmbedding is a simple implementation of the Embedding interface
 * that provides a mapping from strings (categories) to hypervectors.
 *
 * This embedding is useful for representing categorical data, for example order types
 */

public class VanillaEmbedding implements Embedding<String>, Serializable {
    private static final long serialVersionUID = 1L;
    private final Map<String, SparseBinaryVector> hvs;

    public VanillaEmbedding() {
        this.hvs = new HashMap<>();
    }

    @Override
    public SparseBinaryVector forward(String x) {
        return hvs.computeIfAbsent(x, k -> SparseBinaryVector.randVector());
    }

    @Override
    public SparseBinaryVector forward(int level) {
        return null;
    }

    @Override
    public String back(SparseBinaryVector hv) {
        return hvs.entrySet().stream()
                .min(Comparator.comparingInt(e -> hv.hammingDistance(e.getValue())))
                .map(Map.Entry::getKey)
                .orElse(null);
    }

    public static void main(String[] args) {
        VanillaEmbedding nameEmbed = new VanillaEmbedding();
        SparseBinaryVector nameHV = nameEmbed.forward("John");
        System.out.println(nameEmbed.back(nameHV));
    }

}






