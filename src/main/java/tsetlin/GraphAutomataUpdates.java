package tsetlin;

import com.github.kilianB.pcg.sync.PcgRR;

public class GraphAutomataUpdates {

    private final int numberOfNodes;
    private TsetlinAttentionLearning graphAttentionLearning;

    private final int INT_SIZE = 32;
    private final int STATE_BITS = 8;
    private int MESSAGE_CHUNKS;
    private int LA_CHUNKS;
    private int MAX_INCLUDED_LITERALS = 0;
    private boolean BOOST_TRUE_POSITIVE_FEEDBACK = true;

    private int FEATURE_BITS = 0;
    private int CLAUSES;
    private int THRESHOLD;
    private int CLASSES;
    private int Q = 1;
    private int FILTER = 0xffffffff;
    private int MESSAGE_FILTER = 0xffffffff;
    private boolean NEGATIVE_CLAUSES;
    private int nodeChunks;
    private int nodeFilter;
    private int numberLayers = 1;
    private float s;

    PcgRR rng = new PcgRR();

    public GraphAutomataUpdates(TsetlinAttentionLearning graphAttentionLearning) {

        this.graphAttentionLearning = graphAttentionLearning;

        this.LA_CHUNKS = graphAttentionLearning.getLa_chunks();
        this.MESSAGE_CHUNKS = graphAttentionLearning.getLa_chunks();
        this.MAX_INCLUDED_LITERALS = graphAttentionLearning.getMax_literals();
        this.BOOST_TRUE_POSITIVE_FEEDBACK = graphAttentionLearning.isBoost();

        this.FEATURE_BITS = graphAttentionLearning.getFeature_bits();
        this.CLAUSES = graphAttentionLearning.getnClauses();
        this.THRESHOLD = graphAttentionLearning.getThreshold();
        this.CLASSES = graphAttentionLearning.getnClasses();
        this.NEGATIVE_CLAUSES = true;
        this.FILTER = graphAttentionLearning.getFilter();
        this.MESSAGE_FILTER = graphAttentionLearning.getFilter();

        this.numberLayers = graphAttentionLearning.getNumberLayers();
        this.s = graphAttentionLearning.getS();
        this.Q = 1;


        numberOfNodes = graphAttentionLearning.getNumberNodes();
        this.nodeChunks = (numberOfNodes - 1) / INT_SIZE + 1;
        this.nodeFilter = (numberOfNodes % INT_SIZE) != 0
                ? ~(0xffffffff << (numberOfNodes % INT_SIZE))
                : 0xffffffff;

        //print out all the parameters
        System.out.println("GraphAutomataUpdates initialized with:");
        System.out.println("  LA_CHUNKS: " + LA_CHUNKS);
        System.out.println("  MESSAGE_CHUNKS: " + MESSAGE_CHUNKS);
        System.out.println("  MAX_INCLUDED_LITERALS: " + MAX_INCLUDED_LITERALS);
        System.out.println("  BOOST_TRUE_POSITIVE_FEEDBACK: " + BOOST_TRUE_POSITIVE_FEEDBACK);
        System.out.println("  FEATURE_BITS: " + FEATURE_BITS);
        System.out.println("  CLAUSES: " + CLAUSES);
        System.out.println("  THRESHOLD: " + THRESHOLD);
        System.out.println("  CLASSES: " + CLASSES);
        System.out.println("  FILTER: " + Integer.toBinaryString(FILTER));
        System.out.println("  MESSAGE_FILTER: " + Integer.toBinaryString(MESSAGE_FILTER));
        System.out.println("  numberOfNodes: " + numberOfNodes);
        System.out.println("  nodeChunks: " + nodeChunks);
        System.out.println("  nodeFilter: " + Integer.toBinaryString(nodeFilter));
        System.out.println("  s: " + s);
        System.out.println("  numberLayers: " + numberLayers);

    }

    // ----------------------------------------------------------------------------
    // === device‐inline update_clause_message ====================================
    // ----------------------------------------------------------------------------
    public void updateClauseMessage(
            float s,
            int targetSign,
            int[] taState,                   // length == MESSAGE_CHUNKS*STATE_BITS
            boolean clauseOutput,
            int clauseNode,
            int numberOfIncludeActions,
            int[] X                        // dims: [numberOfNodes*MESSAGE_CHUNKS]
    ) {
        if (targetSign > 0) {
            // Type I Feedback
            for (int laChunk = 0; laChunk < MESSAGE_CHUNKS; laChunk++) {
                // generate random bit‐mask
                int laFeedback = 0;
                for (int b = 0; b < INT_SIZE; b++) {
                    if (rng.nextFloat() <= 1.0f / s) {
                        laFeedback |= (1 << b);
                    }
                }

                if (clauseOutput && numberOfIncludeActions <= MAX_INCLUDED_LITERALS) {
                    int includeMask = X[clauseNode*MESSAGE_CHUNKS + laChunk];
                    if (!BOOST_TRUE_POSITIVE_FEEDBACK) {
                        includeMask &= ~laFeedback;
                    }
                    inc(taState, laChunk, includeMask);

                    int excludeMask = (~X[clauseNode*MESSAGE_CHUNKS + laChunk]) & laFeedback;
                    dec(taState, laChunk, excludeMask);
                }
                else {
                    // just negative update
                    dec(taState, laChunk, laFeedback);
                }
            }
        }
        else if (targetSign < 0 && clauseOutput) {
            // Type II Feedback
            for (int laChunk = 0; laChunk < MESSAGE_CHUNKS; laChunk++) {

                int bitmask = (~X[clauseNode*MESSAGE_CHUNKS + laChunk])
                        & (~taState[laChunk*STATE_BITS + STATE_BITS-1]);

                inc(taState, laChunk, bitmask);
            }
        }
    }

    // ----------------------------------------------------------------------------
    // === device‐inline update_clause ============================================
    // (identical to updateClauseMessage but using LA_CHUNKS instead of MESSAGE_CHUNKS)
    // ----------------------------------------------------------------------------
    public void updateClause(
            float s,
            int targetSign,
            int[] taState,                   // length == LA_CHUNKS*STATE_BITS
            boolean clauseOutput,
            int clauseNode,
            int numberOfIncludeActions,
            int[] X                     // length == LA_CHUNKS
    ) {
        if (targetSign > 0) {
            // Type I Feedback
            for (int laChunk = 0; laChunk < LA_CHUNKS; laChunk++) {
                int laFeedback = 0;
                for (int b = 0; b < INT_SIZE; b++) {
                    if (rng.nextFloat() <= 1.0f / s) {
                        laFeedback |= (1 << b);
                    }
                }

                if (clauseOutput && numberOfIncludeActions <= MAX_INCLUDED_LITERALS) {
                    int includeMask = X[clauseNode*LA_CHUNKS + laChunk];
                    if (!BOOST_TRUE_POSITIVE_FEEDBACK) {
                        includeMask &= ~laFeedback;
                    }
                    inc(taState, laChunk, includeMask);

                    int excludeMask = (~X[clauseNode*LA_CHUNKS + laChunk]) & laFeedback;
                    dec(taState, laChunk, excludeMask);
                }
                else {
                    dec(taState, laChunk, laFeedback);
                }
            }
        }
        else if (targetSign < 0 && clauseOutput) {
            // Type II Feedback
            for (int laChunk = 0; laChunk < LA_CHUNKS; laChunk++) {
                int bitmask = (~X[clauseNode*LA_CHUNKS + laChunk])
                        & (~taState[laChunk*STATE_BITS + STATE_BITS-1]);
                inc(taState, laChunk, bitmask);
            }
        }
    }

    // ----------------------------------------------------------------------------
    // === global update_message kernel ===========================================
    // ----------------------------------------------------------------------------
    public  void updateMessage(
            float s,
            int[][] globalTaState,        // dims: [CLAUSES][MESSAGE_CHUNKS]
            int[] clauseNode,               // length == CLAUSES
            int[] numberOfIncludeActions,   // length == CLAUSES
            int[] X,                      // dims: [numberOfNodes][MESSAGE_CHUNKS]
            int[][] classClauseUpdate       // dims: [CLASSES][CLAUSES]
    ) {
        for (int clause = 0; clause < globalTaState.length; clause++) {

            int[] taState = globalTaState[clause];
            boolean output = (clauseNode[clause] != -1);

            for (int classId = 0; classId < classClauseUpdate.length; classId++) {
                int sign = classClauseUpdate[classId][clause];

                updateClauseMessage(s, sign, taState,
                        output,
                        clauseNode[clause],
                        numberOfIncludeActions[clause],
                        X);
            }
        }
    }

    // ----------------------------------------------------------------------------
    // === global update kernel ==================================================
    // ----------------------------------------------------------------------------
    public void update(
            float s,
            int[][] globalTaState,        // dims: [CLAUSES][LA_CHUNKS]
            int[] clauseNode,               // length == CLAUSES
            int[] numberOfIncludeActions,   // length == CLAUSES
            int[] X,                      // dims: [numberOfGraphs][LA_CHUNKS]
            int[][] classClauseUpdate       // dims: [CLASSES][CLAUSES]
    ) {
        // pick out the slice of X for this graph

        for (int clause = 0; clause < globalTaState.length; clause++) {
            int[] taState = globalTaState[clause];
            boolean output = (clauseNode[clause] != -1);

            for (int classId = 0; classId < classClauseUpdate.length; classId++) {
                int sign = classClauseUpdate[classId][clause];
                updateClause(s, sign, taState,
                        output,
                        clauseNode[clause],
                        numberOfIncludeActions[clause],
                        X);
            }
        }
    }



    public void inc(int[] taState, int chunk, int active) {
        int carry, carryNext;
        int id = chunk * STATE_BITS;
        carry = active;

        for (int b = 0; b < STATE_BITS; ++b) {
            if (carry == 0)
                break;

            carryNext = taState[id + b] & carry; // Sets carry bits (overflow) passing on to next bit
            taState[id + b] = taState[id + b] ^ carry; // Performs increments with XOR
            carry = carryNext;
        }

        if (carry > 0) {
            for (int b = 0; b < STATE_BITS; ++b) {
                taState[id + b] |= carry;
            }
        }
    }

    public void dec(int[] taState, int chunk, int active) {
        int carry, carryNext;
        int id = chunk * STATE_BITS;
        carry = active;

        for (int b = 0; b < STATE_BITS; ++b) {
            if (carry == 0)
                break;

            carryNext = (~taState[id + b]) & carry; // Sets carry bits (overflow) passing on to next bit
            taState[id + b] = taState[id + b] ^ carry; // Performs decrements with XOR
            carry = carryNext;
        }

        if (carry > 0) {
            for (int b = 0; b < STATE_BITS; ++b) {
                taState[id + b] &= ~carry;
            }
        }

    }


    // ------------------------------------------------------------------------
    // Sequential version of the `evaluate` kernel
    // ------------------------------------------------------------------------
    public int[] evaluate(
            int[][] globalClauseNodeOutput, // [CLAUSES][nodeChunks]
            int[][] clauseWeights          // [CLASSES][CLAUSES]
    ) {

        int[] classSum = new int[CLASSES];

        for (int clause = 0; clause < CLAUSES; clause++) {
            boolean clauseOutput = false;

            // Check all but last chunk
            for (int k = 0; k < nodeChunks - 1; k++) {
                if (globalClauseNodeOutput[clause][k] != 0) {
                    clauseOutput = true;
                    break;
                }
            }
            // Check last chunk with mask
            if (!clauseOutput
                    && (globalClauseNodeOutput[clause][nodeChunks - 1] & nodeFilter) != 0) {
                clauseOutput = true;
            }

            if (clauseOutput) {
                for (int cls = 0; cls < CLASSES; cls++) {
                    // atomicAdd → simple add in sequential
                    classSum[cls] += clauseWeights[cls][clause];
                }
            }
        }

        // clamp the sum
        return classSum;
    }

    // ------------------------------------------------------------------------
    // Sequential version of the `select_clause_node` kernel
    // ------------------------------------------------------------------------
    public int[] selectClauseNode(
            int[][] globalClauseNodeOutput) // [CLAUSES][nodeChunks]
    {

        int[] clauseNode = new int[CLAUSES];

        for (int clause = 0; clause < CLAUSES; clause++) {
            // collect all nodes where the bit is set
            int[] trueNodes = new int[numberOfNodes];
            int len = 0;

            for (int node = 0; node < numberOfNodes; node++) {
                int chunk = node / INT_SIZE;
                int pos   = node % INT_SIZE;
                if ((globalClauseNodeOutput[clause][chunk] & (1 << pos)) != 0) {
                    trueNodes[len++] = node;
                }
            }

            if (len > 0) {
                clauseNode[clause] = trueNodes[rng.nextInt(len)];
            } else {
                clauseNode[clause] = -1;
            }
        }

        //print clauseNode
//         for(int i = 0; i < CLAUSES; i++) {
//             System.out.println("clauseNode[" + i + "] = " + clauseNode[i]);
//         }


        return clauseNode;
    }

    // ------------------------------------------------------------------------
    // Sequential version of the `select_clause_updates` kernel
    // ------------------------------------------------------------------------

    /**
     * Sequential version of the `select_clause_updates` kernel
     * @param clauseWeights
     * @param classSum
     * @param y
     * @param clauseNode
     * @return int[][] classClauseUpdate
     */
    public int[][] selectClauseUpdates(
            int[][] clauseWeights,        // [CLASSES][CLAUSES]
            int[] classSum,               // [CLASSES]
            int[] y,                    // [examples][CLASSES]
            int[] clauseNode             // [CLAUSES]
    ) {

        int[][] classClauseUpdate = new int[CLASSES][CLAUSES];

        for (int clause = 0; clause < CLAUSES; clause++) {
            for (int cls = 0; cls < CLASSES; cls++) {
                // clamp the sum
                int sum = classSum[cls];
                if (sum > THRESHOLD)      sum = THRESHOLD;
                else if (sum < -THRESHOLD) sum = -THRESHOLD;

                int trueLabel = y[cls];
                //int target    = 1 - 2 * (sum > trueLabel ? 1 : 0);
                //int target    = 1 - 2 * trueLabel;
                int target = 2*trueLabel - 1; // 1 if trueLabel == 1, -1 if trueLabel == 0

                int w      = clauseWeights[cls][clause];
                int sign   = (w >= 0 ? 1 : 0) - (w < 0 ? 1 : 0);

                float p = (1f/(THRESHOLD*2))*(THRESHOLD + target*sum);

                boolean skip = rng.nextFloat() <= p;

                if (skip) {
                    classClauseUpdate[cls][clause] = 0;
                }
                else {
                    int ts = target * sign; // (1 - 2 * trueLabel) * sign;
                    classClauseUpdate[cls][clause] = ts;

                    if (ts > 0 && clauseNode[clause] != -1 && Math.abs(w) < Integer.MAX_VALUE) {
                        clauseWeights[cls][clause] += sign;
                    }
                    else if (ts < 0 && clauseNode[clause] != -1) { //(2*target-1) * sign
                        clauseWeights[cls][clause] -= sign;
                        if (!NEGATIVE_CLAUSES && clauseWeights[cls][clause] < 1) {
                            clauseWeights[cls][clause] = 1;
                        }
                    }
                }
            }
        }
        return classClauseUpdate;
    }



    /**
     * Sequential equivalent of calculate_messages kernel.
     *
     * @param globalTaState            [CLAUSES][LA_CHUNKS * STATE_BITS]
     * @param nodeType                 [graphs][numberOfNodes]
     * @param numberOfNodeTypes
     * @param globalClauseNodeOutput   [CLAUSES][NODE_CHUNKS]  (output)
     * @param X                        [numberOfNodes][LA_CHUNKS]
     */
    public int[] calculateMessages(
            int[][] globalTaState,
            int[] nodeType,
            int numberOfNodeTypes,
            int[][] globalClauseNodeOutput,
            int[] X
    ) {

        int[] numberOfIncludeActions = new int[CLAUSES];

        for (int clause = 0; clause < CLAUSES; clause++) {

            int[] taState = globalTaState[clause];

            for (int nodeChunk = 0; nodeChunk < nodeChunks; nodeChunk++) {

                globalClauseNodeOutput[clause][nodeChunk] = 0;

                if (nodeChunk == 0) {
                    numberOfIncludeActions[clause] = countNumberOfIncludeActions(taState);
                }

                // start with all bits set
                int clauseNodeOutput = ~0;

                // for each bit position in this chunk
                for (int nodePos = 0; nodePos < INT_SIZE && nodeChunk * INT_SIZE + nodePos < numberOfNodes; nodePos++) {

                    int node = nodeChunk * INT_SIZE + nodePos;

                    // check if this node’s type matches clause mod types
                    if (node >= 0) { //if (nodeType[node] == clause % numberOfNodeTypes) {

                        // check all but last LA chunk
                        for (int laChunk = 0; laChunk < LA_CHUNKS - 1; laChunk++) {
                            int highBitMask = taState[laChunk * STATE_BITS + STATE_BITS - 1];
                            if ((highBitMask & X[node*LA_CHUNKS + laChunk]) != highBitMask) {
                                clauseNodeOutput &= ~(1 << nodePos);
                            }
                        }
                        // last LA chunk with FILTER
                        int highBitMask = taState[(LA_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1];
                        if ((highBitMask & X[node*LA_CHUNKS + LA_CHUNKS - 1] & FILTER) //clauseNode*LA_CHUNKS + laChunk
                                != (highBitMask & FILTER)) {
                            clauseNodeOutput &= ~(1 << nodePos);
                        }
                    } else {
                        clauseNodeOutput &= ~(1 << nodePos);
                    }
                }

                // write back, masking off any extra bits in the final chunk
                if (nodeChunk == nodeChunks - 1) {
                    globalClauseNodeOutput[clause][nodeChunk]
                            = clauseNodeOutput & nodeFilter;

                    //print the clause node output as a binary string
                    //System.out.println("clauseNodeOutput (masked): " + Integer.toBinaryString(globalClauseNodeOutput[clause][nodeChunk]));
                } else {
                    globalClauseNodeOutput[clause][nodeChunk]
                            = clauseNodeOutput;
                }
            }
        }
        return numberOfIncludeActions;
    }

    /**
     * Sequential equivalent of calculate_messages_conditional kernel.
     *
     * @param globalTaState                       [CLAUSES][MESSAGE_CHUNKS * STATE_BITS]
     * @param nodeType                            [graphs][numberOfNodes]
     * @param numberOfNodeTypes
     * @param numberOfNodes
     * @param globalClauseNodeInputCondition     [CLAUSES][NODE_CHUNKS]
     * @param globalClauseNodeOutput              [CLAUSES][NODE_CHUNKS] (output)
     * @param numberOfIncludeActions              [CLAUSES]            (in/out)
     * @param X                                   [numberOfNodes][MESSAGE_CHUNKS]
     */
    public void calculateMessagesConditional(
            int[][] globalTaState,
            int[] nodeType,
            int numberOfNodeTypes,
            int numberOfNodes,
            int[][] globalClauseNodeInputCondition,
            int[][] globalClauseNodeOutput,
            int[] numberOfIncludeActions,
            int[] X
    ) {


        for (int clause = 0; clause < CLAUSES; clause++) {

            int[] taState = globalTaState[clause];

            for (int nodeChunk = 0; nodeChunk < nodeChunks; nodeChunk++) {

                if (nodeChunk == 0) {
                    numberOfIncludeActions[clause] += countNumberOfIncludeActions(taState);
                }

                int clauseNodeOutput = ~0;

                for (int nodePos = 0; nodePos < INT_SIZE && nodeChunk * INT_SIZE + nodePos < numberOfNodes; nodePos++) {

                    int node = nodeChunk * INT_SIZE + nodePos;

                    if (node >= 0) { //if (nodeType[node] == clause % numberOfNodeTypes) {

                        for (int laChunk = 0; laChunk < MESSAGE_CHUNKS - 1; laChunk++) {
                            int highBitMask = taState[laChunk * STATE_BITS + STATE_BITS - 1];
                            if ((highBitMask & X[node*MESSAGE_CHUNKS + laChunk]) != highBitMask) {
                                clauseNodeOutput &= ~(1 << nodePos);
                            }
                        }
                        // last MESSAGE_CHUNK with MESSAGE_FILTER
                        int highBitMask = taState[(MESSAGE_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1];
                        if ((highBitMask & X[node*MESSAGE_CHUNKS + MESSAGE_CHUNKS - 1] & MESSAGE_FILTER)
                                != (highBitMask & MESSAGE_FILTER))
                        {
                            clauseNodeOutput &= ~(1 << nodePos);
                        }
                    }
                    else {
                        clauseNodeOutput &= ~(1 << nodePos);
                    }
                }

                // combine with the conditional output and mask final chunk
                int base = globalClauseNodeInputCondition[clause][nodeChunk];
                if (nodeChunk == nodeChunks - 1) {
                    globalClauseNodeOutput[clause][nodeChunk]
                            = base & clauseNodeOutput & nodeFilter;
                } else {
                    globalClauseNodeOutput[clause][nodeChunk]
                            = base & clauseNodeOutput;
                }
            }
        }
    }


    public int tm_action(int[] ta_state, int la) {

        int la_chunk = la / INT_SIZE;
        int chunk_pos = la % INT_SIZE;

        return (ta_state[la_chunk*STATE_BITS + STATE_BITS -1] & (1 << chunk_pos)) != 0 ? 1 : 0;
    }

    public int countNumberOfIncludeActions(int[] ta_state) {

        int number_of_literals = 0;
        for (int k = 0; k < FEATURE_BITS; k++) {
            number_of_literals += tm_action(ta_state, k) + tm_action(ta_state, FEATURE_BITS + k);
        }
        return number_of_literals;
    }


    /**
     * Aggregates the number of times a literal is included in clauses that had a positive output.
     * @param globalClauseNodeOutput
     * @param taState
     * @return int[] of size 2*FEATURE_BITS, where each index corresponds to a literal and the value is the count of how many times it was included.
     */
    public int[][] countIncludedLiterals(int[][] globalClauseNodeOutput, int[][] taState, int[][] clauseWeights,        // [CLASSES][CLAUSES]
                                         int[] classSum) {

        //get index of the highest classSum
        int maxClassIndex = 0;
        for (int cls = 1; cls < CLASSES; cls++) {
            if (classSum[cls] > classSum[maxClassIndex]) {
                maxClassIndex = cls;
            }
        }

        //get the weights for the class with the highest classSum
        int[] weights = clauseWeights[maxClassIndex];

        int[][] includedLiterals = new int[2][FEATURE_BITS]; //0 index for literals, 1 index for negated literals


        for (int clause = 0; clause < CLAUSES; clause++) {
            boolean clauseOutput = false;

            // Check all but last chunk
            for (int k = 0; k < nodeChunks - 1; k++) {
                if (globalClauseNodeOutput[clause][k] != 0) {
                    clauseOutput = true;
                    break;
                }
            }
            // Check last chunk with mask
            if (!clauseOutput
                    && (globalClauseNodeOutput[clause][nodeChunks - 1] & nodeFilter) != 0) {
                clauseOutput = true;
            }

            if (clauseOutput) {

                for (int k = 0; k < FEATURE_BITS; k++) {

                    int laChunk = k / INT_SIZE;
                    int chunkPos = k % INT_SIZE;

                    // Check if the literal is included in the clause
                    if ((taState[clause][laChunk * STATE_BITS + STATE_BITS - 1] & (1 << chunkPos)) != 0) {
                        includedLiterals[0][k] += weights[clause];
                    }

                    laChunk = (FEATURE_BITS + k) / INT_SIZE;
                    chunkPos = (FEATURE_BITS + k) % INT_SIZE;

                    // Check the negated literal
                    if ((taState[clause][laChunk * STATE_BITS + STATE_BITS - 1] & (1 << chunkPos)) != 0) {
                        includedLiterals[1][k] += weights[clause];
                    }
                }
            }
        }

        return includedLiterals;

    }


    /**
     * Sequential version of the `select_clause_updates` kernel
     * @param clauseWeights
     * @param classSum
     * @param y
     * @param clauseNode
     * @return int[][] classClauseUpdate
     */
    public int[][] selectClauseUpdatesRegression(
            int[][] clauseWeights,        // [CLASSES][CLAUSES]
            int[] classSum,               // [CLASSES]
            int[] y,                    // [examples][CLASSES]
            int[] clauseNode             // [CLAUSES]
    ) {

        int[][] classClauseUpdate = new int[CLASSES][CLAUSES];

        for (int clause = 0; clause < CLAUSES; clause++) {
            for (int cls = 0; cls < CLASSES; cls++) {
                // clamp the sum
                int predY = classSum[cls];
                if (predY > THRESHOLD)      predY = THRESHOLD;
                else if (predY < -THRESHOLD) predY = -THRESHOLD;

                int trueLabel = y[cls];

                int prediction_error = predY - trueLabel;

                double update_p = Math.pow(1.0 * prediction_error / THRESHOLD, 2.0);

                //int target    = 1 - 2 * (sum > trueLabel ? 1 : 0);
                //int target    = 1 - 2 * trueLabel;
                int target = 2*trueLabel - 1; // 1 if trueLabel == 1, -1 if trueLabel == 0

                int w      = clauseWeights[cls][clause];
                int sign   = (w >= 0 ? 1 : 0) - (w < 0 ? 1 : 0);

                float p = (1f/(THRESHOLD*2))*(THRESHOLD + target*predY);

                boolean skip = rng.nextFloat() <= p;

                if (skip) {
                    classClauseUpdate[cls][clause] = 0;
                }
                else {
                    int ts = target * sign; // (1 - 2 * trueLabel) * sign;
                    classClauseUpdate[cls][clause] = ts;

                    if (ts > 0 && clauseNode[clause] != -1 && Math.abs(w) < Integer.MAX_VALUE) {
                        clauseWeights[cls][clause] += sign;
                    }
                    else if (ts < 0 && clauseNode[clause] != -1) { //(2*target-1) * sign
                        clauseWeights[cls][clause] -= sign;
                        if (!NEGATIVE_CLAUSES && clauseWeights[cls][clause] < 1) {
                            clauseWeights[cls][clause] = 1;
                        }
                    }
                }
            }
        }
        return classClauseUpdate;
    }

}
