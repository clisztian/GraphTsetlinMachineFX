package util;


import java.util.Arrays;

/**
 * Utility methods for converting between arrays of 64-bit {@code long} values
 * and arrays of 32-bit {@code int} values by splitting/combining their bit representations.
 */
public class BitUtils {

    /**
     * Converts an array of longs into an array of ints by splitting each 64-bit long
     * into two 32-bit ints (low bits first, then high bits).
     *
     * @param longs the input array of {@code long} values
     * @return an {@code int[]} of length {@code longs.length * 2} containing low/high bits
     * @throws NullPointerException if {@code longs} is null
     */
    public static int[] longsToInts(long[] longs) {
        if (longs == null) {
            throw new NullPointerException("Input long array must not be null");
        }
        int[] ints = new int[longs.length * 2];
        for (int i = 0; i < longs.length; i++) {
            long v = longs[i];
            // low 32 bits
            ints[2 * i]     = (int) v;
            // high 32 bits
            ints[2 * i + 1] = (int) (v >>> 32);
        }
        return ints;
    }

    public static int[] encoded(int[] v) {
        //concatenate with original segments (original + negated) and return
        int[] concatenatedSegments = new int[v.length* 2];
        for (int i = 0; i < v.length; i++) {
            concatenatedSegments[i] = v[i];
            concatenatedSegments[v.length + i] = ~v[i];
        }
        return concatenatedSegments;
    }

    /**
     * Converts an array of ints into an array of longs by combining each pair of 32-bit ints
     * into one 64-bit long (low bits first, then high bits).
     *
     * @param ints the input array of {@code int} values; must have even length
     * @return a {@code long[]} of length {@code ints.length / 2}
     * @throws IllegalArgumentException if {@code ints.length} is not even
     * @throws NullPointerException     if {@code ints} is null
     */
    public static long[] intsToLongs(int[] ints) {
        if (ints == null) {
            throw new NullPointerException("Input int array must not be null");
        }
        if ((ints.length & 1) != 0) {
            throw new IllegalArgumentException("Int array length must be even, but was " + ints.length);
        }
        int n = ints.length / 2;
        long[] longs = new long[n];
        for (int i = 0; i < n; i++) {
            int low  = ints[2 * i];
            int high = ints[2 * i + 1];
            longs[i] = ((long) high << 32) | (low & 0xFFFFFFFFL);
        }
        return longs;
    }

    /**
     * Simple validation main: converts a sample long[] to int[] and back, checking equality.
     */
    public static void main(String[] args) {
        long[] sample = new long[] {
                0L,
                -1L,
                0x01234567_89ABCDEFL,
                Long.MIN_VALUE,
                Long.MAX_VALUE
        };
        System.out.println("Original: " + Arrays.toString(sample));
        int[] asInts = longsToInts(sample);
        System.out.println("As ints:  " + Arrays.toString(asInts));
        long[] back = intsToLongs(asInts);
        System.out.println("Back:     " + Arrays.toString(back));
        System.out.println("Test passed: " + Arrays.equals(sample, back));
    }
}

