package tsetlin;

import com.github.kilianB.pcg.sync.PcgRR;

import java.util.HashMap;

/**
 * Fuzzy-logic Tsetlin Machine with Poisson / Negative Binomial frequency learning.
 * Additions:
 *  - FrequencyConfig for Poisson/NB + exposure offset.
 *  - updateFrequencyPoisson(...) training step aligned to Poisson/NB score equations.
 *  - predictMu(...) and predictOccurrenceProb(...) for inference.
 *  - poissonDeviance(...) utility for monitoring.
 */
public class TMRegression {

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



    private int[] sorted_clause_index;
    private double class_probability;
    /** Maps a class from 0 to threshold-1 a popular vote of the clauses */
    private HashMap<Integer, int[]> regression_clause_map;

    /** Frequency modeling configuration */
    public static final class FrequencyConfig {

        // Guarantees exploration even when |residual| is tiny (common with y=0)
        public double baseUpdateProb = 0.15;

        // Focal-like scaling: use |residual|^gamma (gamma=1.0 is linear; >1 amplifies bigger errors)
        public double residualPower = 1.0;

        // Learn on Bernoulli occurrence residual (r = yOcc - pOcc) instead of Poisson/NB count residual
        public boolean useOccurrenceMode = false;

        /** Scale raw fuzzy score to linear predictor z. Default 0.10 keeps z within a sensible range. */
        public double scoreScale = 0.10;
        /** Clip for z before exp to prevent overflow; exp(10) ~ 22k. */
        public double maxZ = 10.0;
        /** Convert residual magnitude to clause update probability. */
        public double alpha = 0.10;
        /** Clip |residual| to avoid huge probabilities. */
        public double maxResidual = 20.0;
        /** Use log(exposure) as offset if true; else multiply mu by exposure externally. */
        public boolean useOffset = true;
        /** Use Negative Binomial score residual instead of Poisson. */
        public boolean useNegativeBinomial = false;
        /** NB dispersion k (>0). Larger k -> closer to Poisson. */
        public double nbK = 2.0;
        /** Minimum exposure to avoid log(0). */
        public double minExposure = 1e-12;
        /** When true, respect maxLiterals cap in Type I (growth). */
        public boolean capClauseGrowth = true;
        /** When residual ~ 0, apply light decay using feedback_to_la. */
        public boolean neutralDecay = true;
        /** If true, increment/decrement clause_weights during feedback. */
        public boolean weightUpdates = false;
    }

    private final FrequencyConfig freqCfg = new FrequencyConfig();

    public FrequencyConfig frequencyConfig() { return freqCfg; }

    public TMRegression(int threshold, int nFeatures, int nClauses, float max_specificity, boolean boost, int LF, int maxLiterals) {

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

        sorted_clause_index = null;
    }

    public TMRegression initialize() {

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
            if (carry == 0) break;
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
            if (carry == 0) break;
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



    private boolean allowGrowth(int clauseIndex) {
        if (!freqCfg.capClauseGrowth || maxLiterals <= 0) return true;
        int includedCount = 0;
        for (int k = 0; k < la_chunks; ++k) {
            int includeMask = ta_state[clauseIndex][k][state_bits - 1];
            if (k == la_chunks - 1) includeMask &= filter;
            includedCount += Integer.bitCount(includeMask);
        }
        return includedCount <= maxLiterals;
    }

//    // ---------------------------------------------------------------------
//    // Poisson / Negative Binomial frequency update with exposure offset
//    // ---------------------------------------------------------------------
//    /**
//     * Train one sample for claims frequency.
//     * @param Xi bitset over la_chunks (2*nFeatures bits), satisfied literals = 1.
//     * @param y observed claim count (>=0)
//     * @param exposure policy-years (>0, can be fractional)
//     * @return predicted mean mu for this sample after computing residual
//     */
//    public double updateFrequencyPoisson(final int[] Xi, final int y, final double exposure) {
//        // 1) Fuzzy TM score -> linear predictor z
//        int raw = scoreFuzzy(Xi); // in [-T, T]
//
//
//        double z = freqCfg.scoreScale * raw;
//        double zWithOffset = z + (freqCfg.useOffset ? Math.log(Math.max(exposure, freqCfg.minExposure)) : 0.0);
//        double zClip = Math.max(-freqCfg.maxZ, Math.min(freqCfg.maxZ, zWithOffset));
//
//        // 2) Mean and residual
//        double mu = Math.exp(zClip);
//        double residual;
//        if (!freqCfg.useNegativeBinomial) {
//            residual = y - mu; // Poisson score residual
//        } else {
//            double denom = 1.0 + mu / Math.max(freqCfg.nbK, 1e-6);
//            residual = (y - mu) / denom; // NB(k)
//        }
//
//        // 3) Convert residual magnitude to update probability
//        double mag = Math.min(Math.abs(residual), freqCfg.maxResidual);
//        double pUpd = Math.min(1.0, freqCfg.alpha * mag);
//
//        System.out.println("y: " + y + " mu: " + mu + " residual: " + residual + " pUpd: " + pUpd);
//        // 4) Residual sign chooses feedback type
//        if (residual > 0) {
//            // -------- Underprediction: Type I (promote) --------
//            for (int j = 0; j < nClauses; j++) {
//                if (rng.nextDouble() > pUpd) continue;
//
//                int r = clauseResidual(Xi, j);
//                if (r > 0) {
//                    if (freqCfg.weightUpdates) clause_weights[j]++;
//                    initialize_random_streams(j);
//                    boolean allowGrowth = allowGrowth(j);
//                    for (int k = 0; k < la_chunks; ++k) {
//                        if (allowGrowth) {
//                            if (boost)    inc(j, k, Xi[k]);
//                            else          inc(j, k, Xi[k] & (~feedback_to_la[k]));
//                        }
//                        // discourage directions contradicting Xi
//                        dec(j, k, (~Xi[k]) & feedback_to_la[k]);
//                    }
//                } else {
//                    // No match: optional gentle shrink
//                    if (freqCfg.neutralDecay) {
//                        initialize_random_streams(j);
//                        for (int k = 0; k < la_chunks; ++k) dec(j, k, feedback_to_la[k]);
//                    }
//                }
//            }
//        } else if (residual < 0) {
//            // -------- Overprediction: Type II (suppress) --------
//            for (int j = 0; j < nClauses; j++) {
//                if (rng.nextDouble() > pUpd) continue;
//
//                int r = clauseResidual(Xi, j);
//                if (r > 0) {
//                    if (freqCfg.weightUpdates  && clause_weights[j] > 1) clause_weights[j]--;
//                    for (int k = 0; k < la_chunks; ++k) {
//                        // Encourage inclusion of literals that would have falsified this clause for Xi
//                        inc(j, k, (~Xi[k]) & (~ta_state[j][k][state_bits - 1]));
//                    }
//                } else if (freqCfg.neutralDecay) {
//                    initialize_random_streams(j);
//                    for (int k = 0; k < la_chunks; ++k) dec(j, k, feedback_to_la[k]);
//                }
//            }
//        } else {
//            // residual == 0 (rare): optional neutral decay
//            if (freqCfg.neutralDecay) {
//                for (int j = 0; j < nClauses; j++) {
//                    if (rng.nextDouble() > pUpd) continue;
//                    initialize_random_streams(j);
//                    for (int k = 0; k < la_chunks; ++k) dec(j, k, feedback_to_la[k]);
//                }
//            }
//        }
//
//        return mu;
//    }

    /** Single-observation Poisson deviance (0*log(0)=0 by convention). */
    public static double poissonDeviance(final int y, final double muInput) {
        double mu = Math.max(muInput, 1e-12);
        if (y == 0) return 2.0 * (0.0 - (0.0 - mu));
        return 2.0 * (y * Math.log(y / mu) - (y - mu));
    }


    public double updateFrequencyPoisson(final int[] Xi, final int y, final double exposure) {
        // 1) Fuzzy TM score -> linear predictor z
        int raw = scoreFuzzy(Xi); // in [-T, T]
        double z = freqCfg.scoreScale * raw;
        double zWithOffset = z + (freqCfg.useOffset ? Math.log(Math.max(exposure, freqCfg.minExposure)) : 0.0);
        double zClip = Math.max(-freqCfg.maxZ, Math.min(freqCfg.maxZ, zWithOffset));

        // 2) Mean and (optionally) occurrence probability
        double mu = Math.exp(zClip);
        double pOcc = 1.0 - Math.exp(-mu);                  // used if useOccurrenceMode
        if (pOcc < 0) pOcc = 0; else if (pOcc > 1) pOcc = 1;

        // 3) Residual
        double residual;
        if (freqCfg.useOccurrenceMode) {
            int yOcc = (y > 0) ? 1 : 0;
            residual = yOcc - pOcc;                         // Bernoulli residual
        } else if (!freqCfg.useNegativeBinomial) {
            residual = y - mu;                              // Poisson score residual
        } else {
            double denom = 1.0 + mu / Math.max(freqCfg.nbK, 1e-6);
            residual = (y - mu) / denom;                    // NB(k)
        }

        // 4) Always-on update probability: base + alpha * |residual|^gamma (clipped)
        double mag = Math.pow(Math.min(Math.abs(residual), freqCfg.maxResidual),
                Math.max(0.5, freqCfg.residualPower)); // 0.5..2 is sensible
        double pUpd = freqCfg.baseUpdateProb + freqCfg.alpha * mag;
        if (pUpd > 1.0) pUpd = 1.0; else if (pUpd < 0.0) pUpd = 0.0;

        // 5) Residual sign chooses feedback type; update each clause with prob pUpd
        if (residual > 0) {
            // -------- Underprediction: Type I (promote) --------
            for (int j = 0; j < nClauses; j++) {
                if (rng.nextDouble() > pUpd) continue;

                int r = clauseResidual(Xi, j);
                if (r > 0) {
                    if (freqCfg.weightUpdates) clause_weights[j]++;
                    initialize_random_streams(j);
                    boolean allowGrowth = allowGrowth(j);
                    for (int k = 0; k < la_chunks; ++k) {
                        if (allowGrowth) {
                            if (boost)    inc(j, k, Xi[k]);
                            else          inc(j, k, Xi[k] & (~feedback_to_la[k]));
                        }
                        // discourage directions contradicting Xi
                        dec(j, k, (~Xi[k]) & feedback_to_la[k]);
                    }
                } else if (freqCfg.neutralDecay) {
                    initialize_random_streams(j);
                    for (int k = 0; k < la_chunks; ++k) dec(j, k, feedback_to_la[k]);
                }
            }
        } else if (residual < 0) {
            // -------- Overprediction: Type II (suppress) --------
            for (int j = 0; j < nClauses; j++) {
                if (rng.nextDouble() > pUpd) continue;

                int r = clauseResidual(Xi, j);
                if (r > 0) {
                    if (freqCfg.weightUpdates  && clause_weights[j] > 1) clause_weights[j]--;
                    for (int k = 0; k < la_chunks; ++k) {
                        // Encourage inclusion of literals that would have falsified this clause for Xi
                        inc(j, k, (~Xi[k]) & (~ta_state[j][k][state_bits - 1]));
                    }
                } else if (freqCfg.neutralDecay) {
                    initialize_random_streams(j);
                    for (int k = 0; k < la_chunks; ++k) dec(j, k, feedback_to_la[k]);
                }
            }
        } else {
            // residual == 0: optional neutral decay to keep clauses dynamic
            if (freqCfg.neutralDecay) {
                for (int j = 0; j < nClauses; j++) {
                    if (rng.nextDouble() > pUpd) continue;
                    initialize_random_streams(j);
                    for (int k = 0; k < la_chunks; ++k) dec(j, k, feedback_to_la[k]);
                }
            }
        }

        return mu;
    }


    // ---------------------------------------------------------------------
    // Inference helpers
    // ---------------------------------------------------------------------
    /** Predict Poisson/NB mean claims given Xi and exposure. */
    public double predictMu(final int[] Xi, final double exposure) {
        int raw = scoreFuzzy(Xi);
        double z = freqCfg.scoreScale * raw;
        double zWithOffset = z + (freqCfg.useOffset ? Math.log(Math.max(exposure, freqCfg.minExposure)) : 0.0);
        double zClip = Math.max(-freqCfg.maxZ, Math.min(freqCfg.maxZ, zWithOffset));
        return Math.exp(zClip);
    }

    /** Predict probability of at least one claim: 1 - exp(-mu). */
    public double predictOccurrenceProb(final int[] Xi, final double exposure) {
        double mu = predictMu(Xi, exposure);
        double p = 1.0 - Math.exp(-mu);
        // clip numeric noise
        if (p < 0.0) p = 0.0; else if (p > 1.0) p = 1.0;
        return p;
    }

    public int[] getClause_weights() {
        return clause_weights;
    }
}

