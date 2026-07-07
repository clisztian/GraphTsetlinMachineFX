package encoding.hypervector;



import util.HV;

import java.util.*;

import static util.HV.RNG;


/**
 * SparseBinaryVector represents a sparse binary vector with a fixed number of segments and bits per segment
 * Used for the basis of the hypervector encoding
 * A SparseBinaryVector is the atomic unit for the vector space,
 * and is used to represent various features in the encoding
 */
public class SparseBinaryVector {

    private long[] segments;
    private static final int S = HV.SEGMENT_SIZE;
    private static final int L = HV.LONG_SIZE;
    public static final int DIMENSION = S * L;
    public static final int K = HV.K;

    private int ordering = 0;



    public SparseBinaryVector(int S, int L, int K) {
        if (L > 64) throw new IllegalArgumentException("Segment length L cannot be greater than 64.");
        if (K > S * L) throw new IllegalArgumentException("K cannot be greater than total number of bits.");
        this.segments = new long[S];
        generateSparseVector();
    }

    public SparseBinaryVector(long[] segments) {
        this.segments = segments;
    }


    public static SparseBinaryVector randVector() {
        return new SparseBinaryVector(HV.SEGMENT_SIZE, HV.LONG_SIZE, HV.K);
    }


    public static SparseBinaryVector fromBooleanVector(boolean[] next) {
        int S = HV.SEGMENT_SIZE;
        int L = HV.LONG_SIZE;
        long[] segments = new long[S];
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < L; j++) {
                if (next[i * L + j]) {
                    segments[i] |= (1L << j);
                }
            }
        }
        return new SparseBinaryVector(segments);
    }


    private void generateSparseVector() {

        Set<Integer> positions = new HashSet<>();
        while (positions.size() < K) {
            positions.add(RNG.nextInt(S * L));
        }
        for (int pos : positions) {
            int segment = pos / L;
            int bit = pos % L;
            segments[segment] |= (1L << bit);
        }
    }

    // Generates a random half of the vector, either top or bottom half
    public static SparseBinaryVector generateRandomHalf(boolean top) {

        int S = HV.SEGMENT_SIZE;
        int L = HV.LONG_SIZE;
        long[] segments = new long[S];

        int startSegment = top ? 0 : S / 2;
        int endSegment = top ? S / 2 : S;

        for (int i = startSegment; i < endSegment; i++) {
            for (int j = 0; j < L; j++) {
                if (RNG.nextDouble() < HV.SPARSITY) {
                    segments[i] |= (1L << j);
                }
            }
        }
        return new SparseBinaryVector(segments);
    }



    public static SparseBinaryVector getZeroVector() {
        long[] segments = new long[HV.SEGMENT_SIZE];
        return new SparseBinaryVector(segments);
    }

    public static SparseBinaryVector withFixedSegments(int startSegment, int endSegment) {

        int S = HV.SEGMENT_SIZE;
        int L = HV.LONG_SIZE;
        int K = HV.K;

        if (startSegment < 0 || endSegment >= S || startSegment > endSegment) {
            throw new IllegalArgumentException("Invalid segment range");
        }

        SparseBinaryVector vec = getZeroVector();
        int usableBits = (endSegment - startSegment + 1) * L;
        if (K > usableBits) {
            throw new IllegalArgumentException("K exceeds number of available bits in fixed segment range");
        }


        Set<Integer> positions = new HashSet<>();
        while (positions.size() < K) {
            int pos = startSegment * L + RNG.nextInt(usableBits);
            positions.add(pos);
        }

        for (int pos : positions) {
            int seg = pos / L;
            int bit = pos % L;
            vec.segments[seg] |= (1L << bit);
        }


        return vec;
    }

    public static SparseBinaryVector oneSegmentHot(int segmentIndex) {
        if (segmentIndex < 0 || segmentIndex >= HV.SEGMENT_SIZE) {
            throw new IllegalArgumentException("Segment index out of bounds");
        }
        long[] segments = new long[HV.SEGMENT_SIZE];

        //set some RNG bits in the specified segment

        for (int i = 0; i < HV.K/3; i++) {
            int bitIndex = RNG.nextInt(HV.LONG_SIZE);
            segments[segmentIndex] |= (1L << bitIndex);
        }

        return new SparseBinaryVector(segments);
    }

    public static SparseBinaryVector multipleSegmentsHot(Set<Integer> segmentIndices) {
        if (segmentIndices.isEmpty()) {
            throw new IllegalArgumentException("Segment indices cannot be empty");
        }
        long[] segments = new long[HV.SEGMENT_SIZE];
        int localK = (int) (segmentIndices.size() * HV.LONG_SIZE * HV.SPARSITY);

        for (int segmentIndex : segmentIndices) {
            if (segmentIndex < 0 || segmentIndex >= HV.SEGMENT_SIZE) {
                throw new IllegalArgumentException("Segment index out of bounds: " + segmentIndex);
            }
            //set some RNG bits in the specified segment
            for (int i = 0; i < localK; i++) {
                int bitIndex = RNG.nextInt(HV.LONG_SIZE);
                segments[segmentIndex] |= (1L << bitIndex);
            }
        }

        return new SparseBinaryVector(segments);
    }


    public static SparseBinaryVector bundle(List<SparseBinaryVector> vectors) {
        if (vectors.isEmpty()) throw new IllegalArgumentException("Empty vector list");
        int S = HV.SEGMENT_SIZE;
        int L = HV.LONG_SIZE;
        int[][] counts = new int[S][L];

        for (SparseBinaryVector vec : vectors) {
            for (int i = 0; i < S; i++) {
                long segment = vec.segments[i];
                for (int j = 0; j < L; j++) {
                    if (((segment >> j) & 1L) == 1L) {
                        counts[i][j]++;
                    }
                }
            }
        }

        List<Integer> allIndices = new ArrayList<>();
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < L; j++) {
                if (counts[i][j] > 0) {
                    allIndices.add(i * L + j);
                }
            }
        }

        // Sort indices by count (descending)
        allIndices.sort((a, b) -> Integer.compare(
                counts[b / L][b % L],
                counts[a / L][a % L]
        ));

        long[] bundledSegments = new long[S];
        for (int i = 0; i < allIndices.size(); i++) {
            int idx = allIndices.get(i);
            bundledSegments[idx / L] |= (1L << (idx % L));
        }

        return new SparseBinaryVector(bundledSegments);
    }

//    public static SparseBinaryVector bundleWeighted(List<SparseBinaryVector> vectors) {
//        int S = HV.SEGMENT_SIZE;
//        int L = HV.LONG_SIZE;
//        int[] bitCounts = new int[S * L];
//
//        // Count how many times each bit is active
//        for (SparseBinaryVector vec : vectors) {
//            for (int i = 0; i < S; i++) {
//                long segment = vec.getSegments()[i];
//                for (int j = 0; j < L; j++) {
//                    if (((segment >> j) & 1L) != 0) {
//                        bitCounts[i * L + j]++;
//                    }
//                }
//            }
//
//        }
//
//        // Pick the top-K active bits
//        int K = HV.K;
//        List<Integer> indices = new ArrayList<>();
//        for (int i = 0; i < bitCounts.length; i++) indices.add(i);
//        indices.sort((a, b) -> Integer.compare(bitCounts[b], bitCounts[a]));
//
//        long[] bundled = new long[S];
//        for (int i = 0; i < K && i < indices.size(); i++) {
//            int idx = indices.get(i);
//            int seg = idx / L;
//            int bit = idx % L;
//            bundled[seg] |= (1L << bit);
//        }
//
//        return new SparseBinaryVector(bundled);
//    }

    public static SparseBinaryVector bundleWeighted(List<SparseBinaryVector> vectors) {
        int S = HV.SEGMENT_SIZE;
        int L = HV.LONG_SIZE;
        int[] bitCounts = new int[S * L];

        // Count active bits
        for (SparseBinaryVector vec : vectors) {
            for (int i = 0; i < S; i++) {
                long segment = vec.getSegments()[i];
                for (int j = 0; j < L; j++) {
                    if (((segment >> j) & 1L) != 0) {
                        bitCounts[i * L + j]++;
                    }
                }
            }
        }

        // Collect only bits that were active (i.e., got at least one vote)
        List<Integer> activeIndices = new ArrayList<>();
        for (int i = 0; i < bitCounts.length; i++) {
            if (bitCounts[i] > 0) {
                activeIndices.add(i);
            }
        }

        // If no active bits, return zero vector
        if (activeIndices.isEmpty()) {
            return SparseBinaryVector.getZeroVector();
        }

        // Sort by number of votes descending
        activeIndices.sort((a, b) -> Integer.compare(bitCounts[b], bitCounts[a]));

        int maxBits = Math.min(HV.K, activeIndices.size());

        long[] bundled = new long[S];
        for (int i = 0; i < maxBits; i++) {
            int idx = activeIndices.get(i);
            int seg = idx / L;
            int bit = idx % L;
            bundled[seg] |= (1L << bit);
        }

        return new SparseBinaryVector(bundled);
    }


    public SparseBinaryVector permute(int shiftAmount) {
        SparseBinaryVector result = SparseBinaryVector.getZeroVector();
        for (int i = 0; i < S * L; i++) {
            int seg = i / L;
            int bit = i % L;
            if ((segments[seg] & (1L << bit)) != 0) {
                int shifted = (i + shiftAmount + S * L) % (S * L);
                int newSeg = shifted / L;
                int newBit = shifted % L;
                result.segments[newSeg] |= (1L << newBit);
            }
        }
        return result;
    }

    public SparseBinaryVector unpermute(int shiftAmount) {
        SparseBinaryVector result = SparseBinaryVector.getZeroVector();

        int dimension = S * L;

        for (int i = 0; i < dimension; i++) {
            int seg = i / L;
            int bit = i % L;

            if ((this.segments[seg] & (1L << bit)) != 0) {
                // Compute the original index before shift
                int originalIndex = (i - shiftAmount + dimension) % dimension;
                int newSeg = originalIndex / L;
                int newBit = originalIndex % L;
                result.segments[newSeg] |= (1L << newBit);
            }
        }

        return result;
    }

    public SparseBinaryVector bind(SparseBinaryVector other) {
        SparseBinaryVector result = SparseBinaryVector.getZeroVector();
        for (int i = 0; i < S; i++) {
            result.segments[i] = this.segments[i] ^ other.segments[i];
        }
        return result;
    }

    public SparseBinaryVector inverseBind(SparseBinaryVector other) {
        return this.bind(other); // XOR is its own inverse
    }

    public SparseBinaryVector bundleWith(SparseBinaryVector other) {
        long[] result = new long[this.segments.length];
        for (int i = 0; i < this.segments.length; i++) {
            result[i] = this.segments[i] | other.segments[i];
        }
        return new SparseBinaryVector(result);
    }

    public SparseBinaryVector sumset(SparseBinaryVector other) {

        int S = HV.SEGMENT_SIZE;
        int L = HV.LONG_SIZE;
        int K = HV.K;

        SparseBinaryVector result = getZeroVector();
        int[] counts = new int[S * L];

        for (int i = 0; i < S * L; i++) {
            int seg = i / L;
            int bit = i % L;
            if ((this.segments[seg] & (1L << bit)) != 0) counts[i]++;
            if ((other.segments[seg] & (1L << bit)) != 0) counts[i]++;
        }

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < counts.length; i++) indices.add(i);
        indices.sort((a, b) -> Integer.compare(counts[b], counts[a]));

        Arrays.fill(result.segments, 0);
        for (int i = 0; i < K && i < indices.size(); i++) {
            int pos = indices.get(i);
            int seg = pos / L;
            int bit = pos % L;
            result.segments[seg] |= (1L << bit);
        }

        return result;
    }

    public static SparseBinaryVector logic_majority(List<SparseBinaryVector> vectors) {
        int[] counts = new int[DIMENSION];
        for (SparseBinaryVector vector : vectors) {
            long[] data = vector.getSegments();
            for (int i = 0; i < S; i++) {
                for (int bit = 0; bit < HV.SIZE; bit++) {
                    if ((data[i] & (1 << bit)) != 0) {
                        counts[i * HV.SIZE + bit]++;
                    }
                }
            }
        }

        long[] majorityData = new long[S];
        int threshold = vectors.size() / 2;
        for (int i = 0; i < S; i++) {
            for (int bit = 0; bit < HV.SIZE; bit++) {
                if (counts[i * HV.SIZE + bit] > threshold) {
                    majorityData[i] |= 1 << bit;
                }
            }
        }
        return new SparseBinaryVector(majorityData);
    }

    public static SparseBinaryVector bundle(List<SparseBinaryVector> vectors, int K) {
        if (vectors.isEmpty()) throw new IllegalArgumentException("Cannot bundle empty vector list.");
        int S = SparseBinaryVector.S;
        int L = SparseBinaryVector.L;
        int[] counts = new int[S * L];
        for (SparseBinaryVector vec : vectors) {
            for (int i = 0; i < S; i++) {
                for (int b = 0; b < L; b++) {
                    if ((vec.segments[i] & (1L << b)) != 0) {
                        counts[i * L + b]++;
                    }
                }
            }
        }
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < counts.length; i++) {
            indices.add(i);
        }
        indices.sort((a, b) -> Integer.compare(counts[b], counts[a]));

        SparseBinaryVector result = new SparseBinaryVector(S, L, K);
        Arrays.fill(result.segments, 0);
        for (int i = 0; i < K && i < indices.size(); i++) {
            int pos = indices.get(i);
            int seg = pos / L;
            int bit = pos % L;
            result.segments[seg] |= (1L << bit);
        }
        return result;
    }

    public int hammingDistance(SparseBinaryVector other) {

        int distance = 0;
        for (int i = 0; i < S; i++) {
            distance += Long.bitCount(this.segments[i] ^ other.segments[i]);
        }
        return distance;
    }

    public double jaccardDistance(SparseBinaryVector other) {
        int intersection = 0, union = 0;
        for (int i = 0; i < segments.length; i++) {
            long a = this.segments[i];
            long b = other.segments[i];
            intersection += Long.bitCount(a & b);
            union += Long.bitCount(a | b);
        }
        return 1.0 - (double) intersection / (union == 0 ? 1 : union);
    }

    public long[] getSegments() {
        return segments;
    }

    public void printVector() {
        System.out.println("Original Segments:");
        for (long segment : segments) {
            System.out.println(String.format("%64s", Long.toBinaryString(segment)).replace(' ', '0'));
        }
    }

    public long[] encoded() {
        //concatenate with original segments (original + negated) and return
        long[] concatenatedSegments = new long[S * 2];
        for (int i = 0; i < S; i++) {
            concatenatedSegments[i] = segments[i];
            concatenatedSegments[S + i] = ~segments[i];
        }
        return concatenatedSegments;
    }

    public static SparseBinaryVector createOrthogonalVector(SparseBinaryVector other) {
        SparseBinaryVector result = new SparseBinaryVector(HV.SEGMENT_SIZE, HV.LONG_SIZE, HV.K);
        for (int i = 0; i < HV.SEGMENT_SIZE; i++) {
            result.segments[i] = ~other.segments[i];
        }
        return result;
    }

    public int[] toBooleanIntArray() {
        int[] booleanArray = new int[S * L];
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < L; j++) {
                booleanArray[i * L + j] = (segments[i] & (1L << j)) != 0 ? 1 : 0;
            }
        }
        return booleanArray;
    }

    public boolean[] toBooleanVector() {
        boolean[] result = new boolean[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            int index = i / L;
            int bit = i % L;
            result[i] = ((segments[index] >>> bit) & 1) == 1;
        }
        return result;
    }

    public void setOrdering(int ordering) {
        this.ordering = ordering;
    }
    public int getOrdering() {
        return ordering;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (long segment : segments) {
            sb.append(String.format("%64s", Long.toBinaryString(segment)).replace(' ', '0')).append("\n");
        }
        return sb.toString();
    }

    public static List<SparseBinaryVector> generateRoleVectors(int numRoles, int candidatesPerRole) {
        List<SparseBinaryVector> roles = new ArrayList<>();

        // Start with a base vector
        roles.add(SparseBinaryVector.randVector());

        while (roles.size() < numRoles) {
            SparseBinaryVector best = null;
            int bestMinDistance = -1;

            for (int i = 0; i < candidatesPerRole; i++) {
                SparseBinaryVector candidate = SparseBinaryVector.randVector();

                int minDistance = roles.stream()
                        .mapToInt(existing -> existing.hammingDistance(candidate))
                        .min()
                        .orElse(SparseBinaryVector.DIMENSION);

                if (minDistance > bestMinDistance) {
                    best = candidate;
                    bestMinDistance = minDistance;
                }
            }

            roles.add(best);
        }

        return roles;
    }

    /**
     * Override equals
     * @param o
     * @return true if the segments are equal, false otherwise
     */

    public boolean equals(SparseBinaryVector o) {
        if (this == o) return true;
        return Arrays.equals(segments, o.segments);
    }


    public static List<SparseBinaryVector> generateSegmentSeparatedRankRoles(int numRanks) {

        int segsPerRank = HV.SEGMENT_SIZE / numRanks;

        if (segsPerRank * numRanks > S) {
            throw new IllegalArgumentException("Not enough segments for the given number of price ranks.");
        }

        List<SparseBinaryVector> result = new ArrayList<>();

        for (int i = 0; i < numRanks; i++) {

            Set<Integer> fixedSegs = new HashSet<>();
            for (int s = 0; s < segsPerRank; s++) {
                fixedSegs.add(i * segsPerRank + s);
            }

            SparseBinaryVector vector = SparseBinaryVector.multipleSegmentsHot(fixedSegs);
            //System.out.println(vector);
            result.add(vector);

        }

        return result;
    }

    /**
     * Creates a SparseBinaryVector from the top-K literals based on their counts.
     * @param literalCounts
     * @return
     */
    public static SparseBinaryVector fromTopLiterals(int[] literalCounts) {
        if (literalCounts.length != SparseBinaryVector.DIMENSION) {
            throw new IllegalArgumentException("Literal count array must be of dimension " + SparseBinaryVector.DIMENSION);
        }

        int K = HV.K; // desired number of bits to keep
        int S = HV.SEGMENT_SIZE;
        int L = HV.LONG_SIZE;

        // Create an index array to sort by count
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < literalCounts.length; i++) {
            indices.add(i);
        }

        // Sort indices by descending count value
        indices.sort((a, b) -> Integer.compare(literalCounts[b], literalCounts[a]));

        // Build long[] segments by setting top-K active bits
        long[] segments = new long[S];
        for (int i = 0; i < K && i < indices.size(); i++) {
            int pos = indices.get(i);
            int seg = pos / L;
            int bit = pos % L;
            segments[seg] |= (1L << bit);
        }




        return new SparseBinaryVector(segments);
    }

    public double percentOverlap(SparseBinaryVector other) {
        int intersection = 0;
        int total = 0;
        for (int i = 0; i < segments.length; i++) {
            long a = this.segments[i];
            long b = other.segments[i];
            intersection += Long.bitCount(a & b);
            total += Long.bitCount(this.segments[i]);
        }
        return (double) intersection / (total == 0 ? 1 : total);
    }
}
