package encoding.embedding;


import encoding.hypervector.SparseBinaryVector;

/** A generic embedding from type T to a VanillaBHV, with invert-back. */
public interface Embedding<T> {
    /** Map a value into its hypervector. */
    SparseBinaryVector forward(T value);

    SparseBinaryVector forward(int level);

    /** Recover the closest value from a hypervector. */
    T         back   (SparseBinaryVector hv);
}