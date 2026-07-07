package tsetlin;

import com.github.kilianB.pcg.sync.PcgRR;

import java.util.HashMap;
public class TMClassifier {

    final int INT_SIZE = 32;

    private int LF;
    private int maxLiterals;

    private int nClauses;
    private int la_chunks;
    private int state_bits;
    private int clause_chunks;
    private int nFeatures;
    private int T;
    private int filter;
    private float max_specificity;
    private float s;
    private boolean boost;

    private int PREDICT;
    private int UPDATE;


    private int[][][] ta_state; //[CLAUSES][LA_CHUNKS][STATE_BITS];
    private int[] clause_output; //[CLAUSE_CHUNKS];
    private int[] feedback_to_la; //[LA_CHUNKS];
    private int[] feedback_to_clauses; //[CLAUSE_CHUNKS];
    private int[] clause_weights;

    PcgRR rng;

    private boolean weighted;

    private int[] sorted_clause_index;
    private double class_probability;
    /**
     * Maps a class from 0 to threshold-1 a popular vote of the clauses
     */
    private HashMap<Integer, int[]> regression_clause_map;




    public TMClassifier(int threshold, int nFeatures, int nClauses, float max_specificity, boolean boost, int LF, int maxLiterals) {

        this.LF = LF;
        this.maxLiterals = maxLiterals;
        this.T = threshold;
        this.nFeatures = nFeatures;
        this.nClauses = nClauses;
        this.max_specificity = max_specificity;
        this.boost = boost;

        this.state_bits = 8;
        this.clause_chunks = (nClauses - 1)/INT_SIZE + 1;
        this.la_chunks = (2*nFeatures - 1)/INT_SIZE + 1;
        this.s = max_specificity;

        PREDICT = 1;
        UPDATE = 0;
        rng = new PcgRR();

        if (((nFeatures*2) % 32) != 0) {
            this.filter  = (~(0xffffffff << ((nFeatures*2) % 32)));
        } else {
            this.filter = 0xffffffff;
        }
        weighted = false;
        sorted_clause_index = null;
    }

    public TMClassifier initialize() {

        ta_state = new int[nClauses][la_chunks][state_bits];
        clause_output = new int[clause_chunks];
        feedback_to_la = new int[la_chunks];
        feedback_to_clauses = new int[clause_chunks];
        clause_weights = new int[nClauses];
        sorted_clause_index = null;


        for (int j = 0; j < nClauses; ++j) {
            for (int k = 0; k < la_chunks; ++k) {
                for (int b = 0; b < state_bits-1; ++b) {
                    ta_state[j][k][b] = ~0;
                }
                ta_state[j][k][state_bits-1] = 0;
            }
            clause_weights[j] = 1;
        }

        return this;
    }


    public void initialize_random_streams(int clause) {

        // Initialize all bits to zero
        feedback_to_la = new int[la_chunks];

        int n = 2 * nFeatures;
        double p = 1f / s;

        if(max_specificity > 0) {
            p = 1.0 / (clause * (max_specificity) / nClauses);
        }
        int active = (int)((n * p * (1 - p))*rng.nextGaussian() + (n * p));

        active = Math.min(active, n);
        active = Math.max(active, 0);

        while (active-- != 0) {

            long rand = Integer.toUnsignedLong(~rng.nextInt());
            int f = (int)(rand % n);

            while ((feedback_to_la[f / 32] & (1 << (f % 32))) != 0) {
                f = (int) (Integer.toUnsignedLong(~rng.nextInt()) % n);
            }
            feedback_to_la[f / 32] |= 1 << (f % 32);
        }
    }



    // ------------------------------------------------------------
    // FPTM: compute fuzzy residual for clause j on sample Xi
    // c = min(LF, #included) - mismatches; mismatches = popcount(included & ~Xi)
    // ------------------------------------------------------------
    private int clauseResidual(final int[] Xi, final int clauseIndex) {
        int includedCount = 0;
        int mismatches = 0;

        for (int k = 0; k < la_chunks; ++k) {
            int includeMask = ta_state[clauseIndex][k][state_bits - 1];
            if (k == la_chunks - 1) includeMask &= filter;

            includedCount += Integer.bitCount(includeMask);

            int satisfiedBits = Xi[k];
            int mismatchBits = includeMask & ~satisfiedBits;
            mismatches += Integer.bitCount(mismatchBits);
        }

        int c = Math.min(includedCount, LF);
        c -= mismatches;
        return Math.max(c, 0);
    }


    /* FPTM: Sum up votes using fuzzy residuals instead of boolean outputs.
       NOTE: signature changed to accept Xi. */
    public int sum_up_class_votes(final int[] Xi) {
        int class_sum = 0;

        int pos_sum_weights = 0;
        int positive_polarity_votes = 0;
        int negative_polarity_votes = 0;
        int neg_sum_weights = 0;

        for (int j = 0; j < nClauses; j++) {
            int residual = clauseResidual(Xi, j);
            int w = clause_weights[j];

            if (j % 2 == 0) { // even = positive polarity
                class_sum += w * residual;
                positive_polarity_votes += w * residual;
                pos_sum_weights += w;
            } else {            // odd = negative polarity
                class_sum -= w * residual;
                negative_polarity_votes += w * residual;
                neg_sum_weights += w;
            }
        }

        // (probability kept in case you use it elsewhere; not used by training here)
        double pos_polar = pos_sum_weights == 0 ? 0.0 : (1.0 * positive_polarity_votes / pos_sum_weights);
        double neg_polar = neg_sum_weights == 0 ? 0.0 : (1.0 * negative_polarity_votes / neg_sum_weights);
        double prob = Math.min(1.0, .50 + pos_polar - neg_polar);
        this.class_probability = prob;

        return class_sum;
    }










    /** Raw fuzzy score: sum(pos residuals) - sum(neg residuals), clamped to [-T, T]. */
    public int scoreFuzzy(int[] Xi) {
        int sum = 0;
        for (int j = 0; j < nClauses; j++) {
            int r = clauseResidual(Xi, j);            // 0..LF
            int w = clause_weights[j];                // keep your weighting
            sum += ((j & 1) == 0 ? +r : -r) * w;      // even = positive, odd = negative
        }
        if (sum > T) sum = T; else if (sum < -T) sum = -T;
        return sum;
    }







    // Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
    private void inc(int clause, int chunk, int active) {

        int carry, carry_next;
        carry = active;
        for (int b = 0; b < state_bits; ++b) {
            if (carry == 0)
                break;

            carry_next = ta_state[clause][chunk][b] & carry; // Sets carry bits (overflow) passing on to next bit
            ta_state[clause][chunk][b] = ta_state[clause][chunk][b] ^ carry; // Performs increments with XOR
            carry = carry_next;
        }

        if (carry > 0) {
            for (int b = 0; b < state_bits; ++b) {
                ta_state[clause][chunk][b] |= carry;
            }
        }
    }

    // Decrement the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
    private void dec(int clause, int chunk, int active) {

        int carry, carry_next;

        carry = active;
        for (int b = 0; b < state_bits; ++b) {
            if (carry == 0)
                break;

            carry_next = (~ta_state[clause][chunk][b]) & carry; // Sets carry bits (overflow) passing on to next bit
            ta_state[clause][chunk][b] = ta_state[clause][chunk][b] ^ carry; // Performs increments with XOR
            carry = carry_next;
        }

        if (carry > 0) {
            for (int b = 0; b < state_bits; ++b) {
                ta_state[clause][chunk][b] &= ~carry;
            }
        }
    }







    // Drop-in: mirrors the Julia feedback! logic using fuzzy clause residuals.
// Requires: clauseResidual(int[] Xi, int j), inc(...), dec(...), initialize_random_streams(j),
//           fields T, nClauses, la_chunks, state_bits, filter, rng, weighted, clause_weights[], maxLiterals.
    public void updateFuzzy(int[] Xi, int target) {
        // 1) Votes & Julia's v
        int class_sum = sum_up_class_votes(Xi); // sum of (pos residuals) - (neg residuals), clamped internally or below
        int v = -class_sum;                     // Julia: v = -(pos - neg)
        if (v > T) v = T; else if (v < -T) v = -T;

        // 2) Per-clause feedback with polarity-specific update probability
        for (int j = 0; j < nClauses; j++) {
            // Even j = positive clause, odd j = negative clause
            boolean positiveSide = ((2 * target - 1) * (1 - 2 * (j & 1))) == 1;

            // Julia: update = (T - v)/(2T) for positive side, (T + v)/(2T) for negative side
            double updateProb = positiveSide ? ((T - v) / (2.0 * T)) : ((T + v) / (2.0 * T));
            if (rng.nextFloat() >= updateProb) continue;

            // Fuzzy match gate: proceed only if clause matches (residual > 0)
            int residual = clauseResidual(Xi, j);

            if (!positiveSide) {
                // ---------------- Type II (negative side in Julia) ----------------
                if (residual > 0) {
                    if (weighted && clause_weights[j] > 1) clause_weights[j]--;
                    for (int k = 0; k < la_chunks; ++k) {
                        // Encourage inclusion of literals that would have falsified the clause
                        inc(j, k, (~Xi[k]) & (~ta_state[j][k][state_bits - 1]));
                    }
                }
            } else {
                // ---------------- Type I (positive side in Julia) ----------------
                initialize_random_streams(j); // prepares feedback_to_la mask

                if (residual > 0) {
                    if (weighted) clause_weights[j]++;

                    // Optional: respect a maxLiterals cap like Julia's L guard
                    boolean allowGrowth = true;
                    if (maxLiterals > 0) {
                        int includedCount = 0;
                        for (int k = 0; k < la_chunks; ++k) {
                            int includeMask = ta_state[j][k][state_bits - 1];
                            if (k == la_chunks - 1) includeMask &= filter;
                            includedCount += Integer.bitCount(includeMask);
                        }
                        allowGrowth = includedCount <= maxLiterals;
                    }

                    for (int k = 0; k < la_chunks; ++k) {
                        // Expand towards satisfied literals (Julia's state++ paths)
                        if (allowGrowth) {
                            if (boost)    inc(j, k, Xi[k]);
                            else          inc(j, k, Xi[k] & (~feedback_to_la[k]));
                        }
                        // Penalize unsatisfied direction (Julia's deterministic dec)
                        dec(j, k, (~Xi[k]) & feedback_to_la[k]);
                    }
                } else {
                    // Clause didn't match: random shrink (Julia: tm.s random decs)

                    for (int k = 0; k < la_chunks; ++k) {
                        dec(j, k, feedback_to_la[k]);
                    }
                }
            }
        }
    }




}
