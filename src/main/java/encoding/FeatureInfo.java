package encoding;


import encoding.embedding.Embedding;
import encoding.hypervector.SparseBinaryVector;

import java.util.Objects;

/**
 * FeatureInfo class represents a feature in a hypervector machine learning context.
 * Main purpose is to encapsulate the name, role, and embedding of a feature, decoding, and
 * similarity computation.
 *
 * @param <T>
 */

public class FeatureInfo<T> {
    public final String       name;   // e.g. “bestBid”, “bestAsk”
    public final SparseBinaryVector role;   // e.g. H_BID
    public final Embedding<T> embed;  // e.g. price level

    public FeatureInfo(String name, SparseBinaryVector role, Embedding<T> embed) {
        this.name  = Objects.requireNonNull(name);
        this.role  = Objects.requireNonNull(role);
        this.embed = Objects.requireNonNull(embed);
    }

    /** Unbinds the feature from a bundled sample HV and returns the decoded value. */
    public T decode(SparseBinaryVector sample) {
        // Unbind
        SparseBinaryVector vh = sample.bind(role);
        // Ask embedding to find the closest level/value
        return embed.back(vh);
    }

    /** Returns per-feature similarity between two samples (0…1). */
    public double similarity(SparseBinaryVector a, SparseBinaryVector b) {
        // Unbind both
        SparseBinaryVector va = a.bind(role);
        SparseBinaryVector vb = b.bind(role);
        // Hamming distance
        int dist = va.hammingDistance(vb);
        return 1.0 - (double) dist / SparseBinaryVector.DIMENSION;
    }
}
