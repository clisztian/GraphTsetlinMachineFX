package tsetlin;

import com.github.kilianB.pcg.sync.PcgRR;
import org.apache.commons.lang3.ArrayUtils;
import util.QuickSort;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * A generalized computational atom for learning a binary input with no group of clauses, as the clauses are shared between
 * other atoms
 *
 * Features:
 *  Generalized convolutional structure for up to 3-dimensional learning
 *  Drop-clause for regularization of clause structure
 *  Integer weighted clauses for faster learning and increased interpretability
 *  The Tsetlin State is shared amongst other clauses
 *  Clause weights are learned according to the current configuration
 *  Output is given by polarization learned by the weights
 *
 * @author lisztian
 *
 */
public class GraphAutomata {


	final int INT_SIZE = 32;
	private final int STATE_BITS = 8;

	private final GraphEncoder encoder;
	private final int number_of_vertices;
	private final int nClauses;
	private final int la_chunks;

	private final int clause_chunks;
	private final int nFeatures;
	private final int T;
	private final int filter;
	private float max_specificity;
	private final float s;
	private final boolean boost;

	private final int PREDICT;
	private final int UPDATE;


	/**
	 * Vertex layer1 Tsetlin Automata
	 */
	private int[][][] ta_state; //[CLAUSES][LA_CHUNKS][STATE_BITS];



	private final int[] clause_output; //[CLAUSE_CHUNKS];
	private final int[] feedback_to_la; //[LA_CHUNKS];
	private int[] feedback_to_clauses; //[CLAUSE_CHUNKS];



	/**
	 * Message Layer Tsetlin Automata
	 */
	private final List<int[][][]> ta_state_message = new ArrayList<>(); //[CLAUSES][LA_CHUNKS][STATE_BITS];
	private final List<int[]> clause_output_message = new ArrayList<>(); //[CLAUSE_CHUNKS];
	private final List<int[]> clause_vertex_message = new ArrayList<>(); //[CLAUSE_CHUNKS];
	private final List<int[]> output_one_vertices_message = new ArrayList<>(); //[LA_CHUNKS];

	private int[][] vertex_clause_map;


	private int[] clause_weights;
	private int[] output_one_vertices;
	private int[] clause_vertex;
	private float[] clause_vertex_coverage;
	private int[] drop_clause;



	private float clause_drop_p = 0;
	private int[] clause_index;
	private double class_probability;
	private int[][] clause_feature_strength;

	private int max_number_literals = 0;
	PcgRR rng;
	private List[] vertex_clause_output;

	/**
	 * Instantiates a general convolutional automata machine
	 * with an encoder, threshold, and other parameters
	 * @param encoder
	 * @param threshold
	 * @param nClauses
	 * @param max_specificity
	 * @param boost
	 */
	public GraphAutomata(GraphEncoder encoder, int threshold, int nClauses, float max_specificity, boolean boost, float clause_drop_p) {

		this.encoder = encoder;
		this.number_of_vertices = encoder.getNumber_of_vertices();
		this.nFeatures = encoder.getFeatureDimension();
		this.T = threshold;

		this.clause_drop_p = clause_drop_p;
		this.nClauses = nClauses;
		this.max_specificity = max_specificity;
		this.boost = boost;

		this.clause_chunks = (nClauses - 1)/INT_SIZE + 1;
		this.la_chunks = encoder.getLayer_ta_chunks();

		this.s = 1f;

		PREDICT = 1;
		UPDATE = 0;
		rng = new PcgRR();

		feedback_to_la = new int[la_chunks];
		clause_output = new int[clause_chunks];

		vertex_clause_output = new List[number_of_vertices];
		//initiate
		for (int i = 0; i < number_of_vertices; i++) {
			vertex_clause_output[i] = new ArrayList<>();
		}

		if (((nFeatures*2) % INT_SIZE) != 0) {
			this.filter  = (~(0xffffffff << ((nFeatures*2) % INT_SIZE)));
		} else {
			this.filter = 0xffffffff;
		}

	}

	public GraphAutomata initializeVertexLayer(int[][][] _ta_state) {

		ta_state = _ta_state;

		drop_clause = new int[clause_chunks];

		feedback_to_clauses = new int[clause_chunks];
		clause_weights = new int[nClauses];
		clause_feature_strength = new int[nClauses][2*nFeatures];

		output_one_vertices = new int[number_of_vertices];
		clause_vertex = new int[nClauses];
		clause_vertex_coverage = new float[nClauses];

		for (int j = 0; j < nClauses; ++j) {
			clause_weights[j] = 1;
		}

		return this;
	}

	public void initializeMessageLayer(int[][][] _ta_state) {

		ta_state_message.add(_ta_state);
		clause_vertex_message.add(new int[nClauses]);
		clause_output_message.add(new int[clause_chunks]);
		output_one_vertices_message.add(new int[number_of_vertices]);

	}




	public void updateDropClause() {

		drop_clause = new int[clause_chunks];

		if(clause_drop_p > 0) {

			for(int j = 0; j < nClauses; j++) {

				int clause_chunk = j / INT_SIZE;
				int clause_chunk_pos = j % INT_SIZE;

				if(rng.nextFloat() < clause_drop_p) {
					drop_clause[clause_chunk]  |= (1 << clause_chunk_pos);
				}
			}
		}
	}


	public void initialize_random_streams(int clause) {


		Arrays.fill(feedback_to_la, 0);

		int n = 2 * nFeatures;
		double p = 1f / s;

		if(max_specificity > 0) {
			p = 1.0 / (s + 1.0 * clause * (max_specificity - s) / nClauses);
		}
		int active = (int)((n * p * (1 - p))*rng.nextGaussian() + (n * p));

		active = Math.min(active, n);
		active = Math.max(active, 0);

		while (active-- != 0) {

			long rand = Integer.toUnsignedLong(~rng.nextInt());
			int f = (int)(rand % n);

			while ((feedback_to_la[f / INT_SIZE] & (1 << (f % INT_SIZE))) != 0) {
				f = (int) (Integer.toUnsignedLong(~rng.nextInt()) % n);
			}
			feedback_to_la[f / INT_SIZE] |= 1 << (f % INT_SIZE);
		}
	}


	// Increment the states of each of those INT_SIZE Tsetlin Automata flagged in the active bit vector.
	private void inc(int clause, int chunk, int active) {

		int carry, carry_next;
		carry = active;
		for (int b = 0; b < STATE_BITS; ++b) {
			if (carry == 0)
				break;

			carry_next = ta_state[clause][chunk][b] & carry; // Sets carry bits (overflow) passing on to next bit
			ta_state[clause][chunk][b] = ta_state[clause][chunk][b] ^ carry; // Performs increments with XOR
			carry = carry_next;
		}

		if (carry > 0) {
			for (int b = 0; b < STATE_BITS; ++b) {
				ta_state[clause][chunk][b] |= carry;
			}
		}
	}



	// Decrement the states of each of those INT_SIZE Tsetlin Automata flagged in the active bit vector.
	private void dec(int clause, int chunk, int active) {

		int carry, carry_next;

		carry = active;
		for (int b = 0; b < STATE_BITS; ++b) {
			if (carry == 0)
				break;

			carry_next = (~ta_state[clause][chunk][b]) & carry; // Sets carry bits (overflow) passing on to next bit
			ta_state[clause][chunk][b] = ta_state[clause][chunk][b] ^ carry; // Performs increments with XOR
			carry = carry_next;
		}

		if (carry > 0) {
			for (int b = 0; b < STATE_BITS; ++b) {
				ta_state[clause][chunk][b] &= ~carry;
			}
		}
	}


	// Increment the states of each of those INT_SIZE Tsetlin Automata flagged in the active bit vector.
	private void inc(int[][][] _ta_state, int clause, int chunk, int active) {

		int carry, carry_next;
		carry = active;
		for (int b = 0; b < STATE_BITS; ++b) {
			if (carry == 0)
				break;

			carry_next = _ta_state[clause][chunk][b] & carry; // Sets carry bits (overflow) passing on to next bit
			_ta_state[clause][chunk][b] = _ta_state[clause][chunk][b] ^ carry; // Performs increments with XOR
			carry = carry_next;
		}

		if (carry > 0) {
			for (int b = 0; b < STATE_BITS; ++b) {
				_ta_state[clause][chunk][b] |= carry;
			}
		}
	}



	// Decrement the states of each of those INT_SIZE Tsetlin Automata flagged in the active bit vector.
	private void dec(int[][][] _ta_state, int clause, int chunk, int active) {

		int carry, carry_next;

		carry = active;
		for (int b = 0; b < STATE_BITS; ++b) {
			if (carry == 0)
				break;

			carry_next = (~_ta_state[clause][chunk][b]) & carry; // Sets carry bits (overflow) passing on to next bit
			_ta_state[clause][chunk][b] = _ta_state[clause][chunk][b] ^ carry; // Performs increments with XOR
			carry = carry_next;
		}

		if (carry > 0) {
			for (int b = 0; b < STATE_BITS; ++b) {
				_ta_state[clause][chunk][b] &= ~carry;
			}
		}
	}



	/* Sum up the votes for each class */
	public int sum_up_class_votes() {

		int class_sum = 0;

		int pos_sum_weights = 0;
		int positive_polarity_votes = 0;
		int negative_polarity_votes = 0;
		int neg_sum_weights = 0;

		for (int j = 0; j < nClauses; j++) {

			int clause_chunk = j / INT_SIZE;
			int clause_pos = j % INT_SIZE;

			if (clause_weights[j] > 0) {
				class_sum += clause_weights[j] * ((clause_output[clause_chunk] & (1 << clause_pos)) != 0 ? 1 : 0);

				positive_polarity_votes += clause_weights[j] * ((clause_output[clause_chunk] & (1 << clause_pos)) != 0 ? 1 : 0);
				pos_sum_weights += clause_weights[j];

			} else {
				class_sum += clause_weights[j] * ((clause_output[clause_chunk] & (1 << clause_pos)) != 0 ? 1 : 0);

				negative_polarity_votes += clause_weights[j] * ((clause_output[clause_chunk] & (1 << clause_pos)) != 0 ? 1 : 0);
				neg_sum_weights += clause_weights[j];

			}
		}

		class_sum = Math.min(class_sum, T);
		class_sum = Math.max(class_sum, -T);

		double pos_polar = 1.0*positive_polarity_votes/(1.0*pos_sum_weights);
		double neg_polar = 1.0*negative_polarity_votes/(1.0*neg_sum_weights);

		double prob = Math.min(1.0, .50 + pos_polar - neg_polar);

		setClass_probability(prob);

		return class_sum;
	}

	/* Sum up the votes for each class */
	public int sum_up_class_votes_graph() {

		int class_sum = 0;

		int pos_sum_weights = 0;
		int positive_polarity_votes = 0;
		int negative_polarity_votes = 0;
		int neg_sum_weights = 0;

		for (int j = 0; j < nClauses; j++) {

			int clause_chunk = j / INT_SIZE;
			int clause_pos = j % INT_SIZE;

			if (clause_weights[j] > 0) {
				class_sum += clause_weights[j] * ((compositeClauseOutput(clause_chunk) & (1 << clause_pos)) != 0 ? 1 : 0);

				positive_polarity_votes += clause_weights[j] * ((compositeClauseOutput(clause_chunk) & (1 << clause_pos)) != 0 ? 1 : 0);
				pos_sum_weights += clause_weights[j];

			} else {
				class_sum += clause_weights[j] * ((compositeClauseOutput(clause_chunk) & (1 << clause_pos)) != 0 ? 1 : 0);

				negative_polarity_votes += clause_weights[j] * ((compositeClauseOutput(clause_chunk) & (1 << clause_pos)) != 0 ? 1 : 0);
				neg_sum_weights += clause_weights[j];

			}
		}

		class_sum = Math.min(class_sum, T);
		class_sum = Math.max(class_sum, -T);

		double pos_polar = 1.0*positive_polarity_votes/(1.0*pos_sum_weights);
		double neg_polar = 1.0*negative_polarity_votes/(1.0*neg_sum_weights);

		double prob = Math.min(1.0, .50 + pos_polar - neg_polar);

		setClass_probability(prob);

		return class_sum;
	}

	public int getClassSumGraph(int[][] clause_vertex_map) {

		int class_sum = 0;

		for (int j = 0; j < nClauses; j++) {

			//is there at least one vertex in the clause j
			for (int i = 0; i < number_of_vertices; i++) {
				if (clause_vertex_map[i][j] == 1) {
					class_sum += clause_weights[j];
					break;
				}
			}
		}

		class_sum = Math.min(class_sum, T);
		class_sum = Math.max(class_sum, -T);

		return class_sum;
	}





	/* Calculate the output of each clause using the actions of each Tsetline Automaton. */
	public void calculate_clause_output(int[] Xi, int predict) {

		int output_one_patches_count = 0;
		Arrays.fill(clause_output, 0);

		for (int j = 0; j < nClauses; j++) {

			int clause_chunk = j / INT_SIZE;
			int clause_chunk_pos = j % INT_SIZE;

			if(predict == UPDATE && (drop_clause[clause_chunk] & (1 << clause_chunk_pos)) != 0) {
				continue;
			}

			output_one_patches_count = 0;
			clause_vertex_coverage[j] = 0f;

			for(int vertex = 0; vertex < number_of_vertices; ++vertex) {

				int output = 1;
				int all_exclude = 1;

				for (int k = 0; k < la_chunks-1; k++) {
					output = (output == 1) && ((ta_state[j][k][STATE_BITS -1] & Xi[vertex*la_chunks + k]) == ta_state[j][k][STATE_BITS -1]) ? 1 : 0;

					if (output == 0) {
						break;
					}
					all_exclude = (all_exclude == 1) && (ta_state[j][k][STATE_BITS -1] == 0) ? 1 : 0;

				}

				output = (output == 1) && ((ta_state[j][la_chunks-1][STATE_BITS -1] & Xi[vertex*la_chunks + la_chunks-1] & filter) ==
						(ta_state[j][la_chunks-1][STATE_BITS -1] & filter)) ? 1 : 0;

				all_exclude = (all_exclude == 1) && ((ta_state[j][la_chunks-1][STATE_BITS -1] & filter) == 0) ? 1 : 0;

				output = (output == 1) && !(predict == PREDICT && all_exclude == 1) ? 1 : 0;


				if (output == 1) {
					output_one_vertices[output_one_patches_count] = vertex;
					output_one_patches_count++;
				}
			}

			if (output_one_patches_count > 0) {

				clause_output[clause_chunk] |= (1 << clause_chunk_pos);

				int patch_id = (int) (Integer.toUnsignedLong(~rng.nextInt()) % output_one_patches_count);
				clause_vertex[j] = output_one_vertices[patch_id];
				clause_vertex_coverage[j] = 1f*output_one_patches_count/ number_of_vertices;
			}

		}

	}

	/*
	Clause output for graph attention learning
	The vertex_clause_map is a map of vertex to clause and is used to govern which
	vertices get updated
	 */
	public void calculate_clause_output(int[] Xi, int[][][] ta_state, int[] clause_output, int[] clause_vertex, int[] output_one_vertices, int predict) {

		int output_one_patches_count = 0;
		Arrays.fill(clause_output, 0);


		for (int j = 0; j < nClauses; j++) {

			int clause_chunk = j / INT_SIZE;
			int clause_chunk_pos = j % INT_SIZE;

			if(predict == UPDATE && (drop_clause[clause_chunk] & (1 << clause_chunk_pos)) != 0) {
				continue;
			}

			output_one_patches_count = 0;
			clause_vertex_coverage[j] = 0f;

			for(int vertex = 0; vertex < number_of_vertices; ++vertex) {

				int output = 1;
				int all_exclude = 1;

				for (int k = 0; k < la_chunks-1; k++) {
					output = (output == 1) && ((ta_state[j][k][STATE_BITS -1]
							& Xi[vertex*la_chunks + k])
							== ta_state[j][k][STATE_BITS -1]) ? 1 : 0;

					if (output == 0) {
						break;
					}
					all_exclude = (all_exclude == 1) && (ta_state[j][k][STATE_BITS -1] == 0) ? 1 : 0;

				}

				output = (output == 1) && ((ta_state[j][la_chunks-1][STATE_BITS -1] & Xi[vertex*la_chunks + la_chunks-1] & filter) ==
						(ta_state[j][la_chunks-1][STATE_BITS -1] & filter)) ? 1 : 0;

				all_exclude = (all_exclude == 1) && ((ta_state[j][la_chunks-1][STATE_BITS -1] & filter) == 0) ? 1 : 0;

				output = (output == 1) && !(predict == PREDICT && all_exclude == 1) ? 1 : 0;


				if (output == 1) {
					output_one_vertices[output_one_patches_count] = vertex;
					output_one_patches_count++;
				}
			}

			if (output_one_patches_count > 0) {

				clause_output[clause_chunk] |= (1 << clause_chunk_pos);

				int patch_id = (int) (Integer.toUnsignedLong(~rng.nextInt()) % output_one_patches_count);
				clause_vertex[j] = output_one_vertices[patch_id];
				clause_vertex_coverage[j] = 1f*output_one_patches_count/ number_of_vertices;
			}

		}

	}





	// The Tsetlin Machine can be trained incrementally, one training example at a time.
	// Use this method directly for online and incremental training.

	public void update(int[] Xi, int target) {

		calculate_clause_output(Xi, UPDATE);
		int class_sum = sum_up_class_votes();

		initialize_random_streams(rng.nextInt(nClauses));

		float p = (1f/(T*2))*(T + (1 - 2*target)*class_sum);
		feedback_to_clauses = new int[clause_chunks];


		for (int j = 0; j < nClauses; j++) {

			int clause_chunk = j / INT_SIZE;
			int clause_chunk_pos = j % INT_SIZE;

			feedback_to_clauses[clause_chunk] |= (rng.nextFloat() <= p ? 1 : 0) << clause_chunk_pos;
		}

		for (int j = 0; j < nClauses; j++) {

			int sign = clause_weights[j] > 0 ? 1 : -1;

			int clause_chunk = j / INT_SIZE;
			int clause_chunk_pos = j % INT_SIZE;

			if ( ((feedback_to_clauses[clause_chunk] & (1 << clause_chunk_pos) ) == 0) || (drop_clause[clause_chunk] & (1 << clause_chunk_pos)) != 0 ) {
				continue;
			}

			//signs disagree
			if ((2*target-1) * sign == -1 && (clause_output[clause_chunk] & (1 << clause_chunk_pos)) != 0) {

				// Type II Feedback
				clause_weights[j] -= sign;
				for (int k = 0; k < la_chunks; ++k) {

					inc(j, k, (~Xi[clause_vertex[j]*la_chunks + k]) & (~ta_state[j][k][STATE_BITS -1]));
				}

			}

			//signs agree
			else if ((2*target-1) * sign == 1) {
				// Type I Feedback

				//initialize_random_streams(j);

				if ((clause_output[clause_chunk] & (1 << clause_chunk_pos)) != 0) {

					if (max_number_literals == 0 || getNumberOfLiteralsOfClause(j) < max_number_literals) {

						clause_weights[j] += sign;

						for (int k = 0; k < la_chunks; ++k) {

							if (boost)
								inc(j, k, Xi[clause_vertex[j] * la_chunks + k]);
							else
								inc(j, k, Xi[clause_vertex[j] * la_chunks + k] & (~feedback_to_la[k]));

							dec(j, k, (~Xi[clause_vertex[j] * la_chunks + k]) & feedback_to_la[k]);
						}
					}
				}
				else {
					for (int k = 0; k < la_chunks; ++k) {
						dec(j, k, feedback_to_la[k]);
					}
				}
			}
		}
	}

	public int[] updateWeights(int[] clause_output, int target) {

		int[] target_sign = new int[nClauses];
		for (int j = 0; j < nClauses; j++) {

			int sign = clause_weights[j] > 0 ? 1 : -1;

			int clause_chunk = j / INT_SIZE;
			int clause_chunk_pos = j % INT_SIZE;

			//signs disagree
			if ((2*target-1) * sign == -1 && (clause_output[clause_chunk] & (1 << clause_chunk_pos)) != 0) {
				// Type II Feedback
				clause_weights[j] -= sign;
				target_sign[j] = -1;
			}
			else if ((2*target-1) * sign == 1) {
				if ((clause_output[clause_chunk] & (1 << clause_chunk_pos)) != 0) {
					if (max_number_literals == 0 || getNumberOfLiteralsOfClause(ta_state, j) < max_number_literals) {
						clause_weights[j] += sign;
					}
				}
				target_sign[j] = 1;
			}
		}
		return target_sign;

	}

	public void update(int[] Xi, int[][][] ta_state, int[] clause_output, int[] clause_vertex, int class_sum, int target) {


		initialize_random_streams(rng.nextInt(nClauses));

		float p = (1f/(T*2))*(T + (1 - 2*target)*class_sum);
		feedback_to_clauses = new int[clause_chunks];


		for (int j = 0; j < nClauses; j++) {

			int clause_chunk = j / INT_SIZE;
			int clause_chunk_pos = j % INT_SIZE;

			feedback_to_clauses[clause_chunk] |= (rng.nextFloat() <= p ? 1 : 0) << clause_chunk_pos;
		}

		for (int j = 0; j < nClauses; j++) {

			int sign = clause_weights[j] > 0 ? 1 : -1;

			int clause_chunk = j / INT_SIZE;
			int clause_chunk_pos = j % INT_SIZE;

			if ( ((feedback_to_clauses[clause_chunk] & (1 << clause_chunk_pos) ) == 0) || (drop_clause[clause_chunk] & (1 << clause_chunk_pos)) != 0 ) {
				continue;
			}

			//signs disagree
			if ((2*target-1) * sign == -1 && (clause_output[clause_chunk] & (1 << clause_chunk_pos)) != 0) {

				// Type II Feedback
				clause_weights[j] -= sign;
				for (int k = 0; k < la_chunks; ++k) {

					inc(ta_state, j, k, (~Xi[clause_vertex[j]*la_chunks + k]) & (~ta_state[j][k][STATE_BITS -1]));
				}

			}

			//signs agree
			else if ((2*target-1) * sign == 1) {
				// Type I Feedback

				//initialize_random_streams(j);

				if ((clause_output[clause_chunk] & (1 << clause_chunk_pos)) != 0) {

					if (max_number_literals == 0 || getNumberOfLiteralsOfClause(ta_state, j) < max_number_literals) {

						clause_weights[j] += sign;

						for (int k = 0; k < la_chunks; ++k) {

							if (boost)
								inc(ta_state,j, k, Xi[clause_vertex[j] * la_chunks + k]);
							else
								inc(ta_state,j, k, Xi[clause_vertex[j] * la_chunks + k] & (~feedback_to_la[k]));

							dec(ta_state, j, k, (~Xi[clause_vertex[j] * la_chunks + k]) & feedback_to_la[k]);
						}
					}
				}
				else {
					for (int k = 0; k < la_chunks; ++k) {
						dec(ta_state,j, k, feedback_to_la[k]);
					}
				}
			}
		}
	}

	public void update(int[] Xi, int[][][] ta_state, int[] clause_output, int[] clause_vertex, int class_sum, int target, int layer) {


		initialize_random_streams(rng.nextInt(nClauses));

		float p = (1f/(T*2))*(T + (1 - 2*target)*class_sum);
		feedback_to_clauses = new int[clause_chunks];


		for (int j = 0; j < nClauses; j++) {

			int clause_chunk = j / INT_SIZE;
			int clause_chunk_pos = j % INT_SIZE;

			feedback_to_clauses[clause_chunk] |= (rng.nextFloat() <= p ? 1 : 0) << clause_chunk_pos;
		}

		for (int j = 0; j < nClauses; j++) {

			int sign = clause_weights[j] > 0 ? 1 : -1;

			int clause_chunk = j / INT_SIZE;
			int clause_chunk_pos = j % INT_SIZE;

			if ( ((feedback_to_clauses[clause_chunk] & (1 << clause_chunk_pos) ) == 0) || (drop_clause[clause_chunk] & (1 << clause_chunk_pos)) != 0 ) {
				continue;
			}

			//signs disagree
			if ((2*target-1) * sign == -1 && (clause_output[clause_chunk] & (1 << clause_chunk_pos)) != 0) {

				// Type II Feedback
				if(layer == 0)
					clause_weights[j] -= sign;

				for (int k = 0; k < la_chunks; ++k) {

					inc(ta_state, j, k, (~Xi[clause_vertex[j]*la_chunks + k]) & (~ta_state[j][k][STATE_BITS -1]));
				}

			}

			//signs agree
			else if ((2*target-1) * sign == 1) {
				// Type I Feedback

				//initialize_random_streams(j);

				if ((clause_output[clause_chunk] & (1 << clause_chunk_pos)) != 0) {

					if (max_number_literals == 0 || getNumberOfLiteralsOfClause(ta_state, j) < max_number_literals) {

						if(layer == 0)
							clause_weights[j] += sign;

						for (int k = 0; k < la_chunks; ++k) {

							if (boost)
								inc(ta_state,j, k, Xi[clause_vertex[j] * la_chunks + k]);
							else
								inc(ta_state,j, k, Xi[clause_vertex[j] * la_chunks + k] & (~feedback_to_la[k]));

							dec(ta_state, j, k, (~Xi[clause_vertex[j] * la_chunks + k]) & feedback_to_la[k]);
						}
					}
				}
				else {
					for (int k = 0; k < la_chunks; ++k) {
						dec(ta_state,j, k, feedback_to_la[k]);
					}
				}
			}
		}
	}

	public void updateLayer(int[] Xi, int[][][] ta_state, int[] clause_output, int[] clause_vertex, int[] class_clause_sign, int class_sum, int target) {


		initialize_random_streams(rng.nextInt(nClauses));

		float p = (1f/(T*2))*(T + (1 - 2*target)*class_sum);
		feedback_to_clauses = new int[clause_chunks];


		for (int j = 0; j < nClauses; j++) {

			int clause_chunk = j / INT_SIZE;
			int clause_chunk_pos = j % INT_SIZE;

			feedback_to_clauses[clause_chunk] |= (rng.nextFloat() <= p ? 1 : 0) << clause_chunk_pos;
		}

		for (int j = 0; j < nClauses; j++) {


			int clause_chunk = j / INT_SIZE;
			int clause_chunk_pos = j % INT_SIZE;

			if ( ((feedback_to_clauses[clause_chunk] & (1 << clause_chunk_pos) ) == 0) || (drop_clause[clause_chunk] & (1 << clause_chunk_pos)) != 0 ) {
				continue;
			}

			//signs disagree
			if (class_clause_sign[j] == -1 && (clause_output[clause_chunk] & (1 << clause_chunk_pos)) != 0) {
				for (int k = 0; k < la_chunks; ++k) {
					inc(ta_state, j, k, (~Xi[clause_vertex[j]*la_chunks + k]) & (~ta_state[j][k][STATE_BITS -1]));
				}
			}

			//signs agree
			else if (class_clause_sign[j] == 1) {
				// Type I Feedback

				//initialize_random_streams(j);

				if ((clause_output[clause_chunk] & (1 << clause_chunk_pos)) != 0) {
					if (max_number_literals == 0 || getNumberOfLiteralsOfClause(ta_state, j) < max_number_literals) {
						for (int k = 0; k < la_chunks; ++k) {

							if (boost)
								inc(ta_state,j, k, Xi[clause_vertex[j] * la_chunks + k]);
							else
								inc(ta_state,j, k, Xi[clause_vertex[j] * la_chunks + k] & (~feedback_to_la[k]));

							dec(ta_state, j, k, (~Xi[clause_vertex[j] * la_chunks + k]) & feedback_to_la[k]);
						}
					}
				}
				else {
					for (int k = 0; k < la_chunks; ++k) {
						dec(ta_state,j, k, feedback_to_la[k]);
					}
				}
			}
		}
	}



	/**
	 * Update the graph automata given the input Xi for all layers
	 * @param Xi
	 * @param target
	 */
	public void updateGraph(int[][] Xi, int target) {

		calculate_clause_output(Xi[0],
				ta_state, clause_output, clause_vertex, output_one_vertices, UPDATE);

		for (int layer = 0; layer < ta_state_message.size(); layer++) {
			//calculate the clause output for message layer (layer 2)
			calculate_clause_output(Xi[layer + 1],
					ta_state_message.get(layer), clause_output_message.get(layer), clause_vertex_message.get(layer),
					output_one_vertices_message.get(layer), UPDATE);
		}
		int class_sum = sum_up_class_votes_graph();

		//vertex layer fitGraph
		update(Xi[0], ta_state, clause_output, clause_vertex, class_sum, target, 0);

		for(int i = 0; i < nClauses; i++) {
			System.out.println(i + " " + getNumberOfLiteralsOfClause(i));
		}

		//print clause_output
		for (int i = 0; i < clause_output.length; i++) {
			System.out.println("clause_output[" + i + "] = " + Integer.toBinaryString(clause_output[i]));
		}


		//now fitGraph the rest of the layers
		for(int layer = 0; layer < ta_state_message.size(); layer++) {
			//calculate the clause output for message layer (layer 2)
			update(Xi[layer + 1], ta_state_message.get(layer), clause_output_message.get(layer), clause_vertex_message.get(layer), class_sum, target, layer+1);
		}
	}

	public void updateGraph(int[][] Xi, int target, int class_sum) {


		calculate_clause_output(Xi[0],
				ta_state, clause_output, clause_vertex, output_one_vertices, UPDATE);


		int[] target_signs = updateWeights(clause_output, target);

		updateLayer(Xi[0], ta_state, clause_output, clause_vertex, target_signs, class_sum, target);


		for (int layer = 0; layer < ta_state_message.size(); layer++) {
			//calculate the clause output for message layer (layer 2)
			calculate_clause_output(Xi[layer + 1],
					ta_state_message.get(layer), clause_output_message.get(layer), clause_vertex_message.get(layer),
					output_one_vertices_message.get(layer), UPDATE);

			updateLayer(Xi[layer + 1], ta_state_message.get(layer), clause_output_message.get(layer), clause_vertex_message.get(layer), target_signs, class_sum, target);
		}

	}




	int get_state(int clause, int la) {

		int la_chunk = la / INT_SIZE;
		int chunk_pos = la % INT_SIZE;

		int state = 0;
		for (int b = 0; b < STATE_BITS; ++b) {
			if ((ta_state[clause][la_chunk][b] & (1 << chunk_pos)) != 0 ) {
				state |= 1 << b;
			}
		}

		return state;
	}

	public int tm_action(int clause, int la) {

		int la_chunk = la / INT_SIZE;
		int chunk_pos = la % INT_SIZE;

		return (ta_state[clause][la_chunk][STATE_BITS -1] & (1 << chunk_pos)) != 0 ? 1 : 0;
	}

	public int tm_action(int[][][] ta_state, int clause, int la) {

		int la_chunk = la / INT_SIZE;
		int chunk_pos = la % INT_SIZE;

		return (ta_state[clause][la_chunk][STATE_BITS -1] & (1 << chunk_pos)) != 0 ? 1 : 0;
	}



	/**
	 * Returns the number of literals for the given clause.
	 * @param clause_number
	 * @return
	 */
	public int getNumberOfLiteralsOfClause(int clause_number) {

		int number_of_literals = 0;
		for (int k = 0; k < nFeatures; k++) {
			number_of_literals += tm_action(clause_number, k) + tm_action(clause_number, nFeatures + k);
		}
		return number_of_literals;
	}

	public int getNumberOfLiteralsOfClause(int[][][] ta_state, int clause_number) {

		int number_of_literals = 0;
		for (int k = 0; k < nFeatures; k++) {
			number_of_literals += tm_action(ta_state, clause_number, k) + tm_action(ta_state, clause_number, nFeatures + k);
		}
		return number_of_literals;
	}

	public float getSumOfWeights() {

		int sum = 0;
		for (int i = 0; i < nClauses; i++) {

			int clause_chunk = i/ INT_SIZE;
			int clause_chunk_pos = i % INT_SIZE;

			sum += ((clause_output[clause_chunk] & (1 << clause_chunk_pos)) ) * clause_weights[i];
		}
		return  (1f*T + 1f*Math.min(T, Math.max(-T, sum)))/(2f*T);
	}


	/**
	 * Score the graph automata given the input Xi for all layers
	 * @param Xi
	 * @return
	 */
	public int score(int[] Xi) {

		calculate_clause_output(Xi, PREDICT);

		return sum_up_class_votes();
	}

	/**
	 * Score the graph automata given the input Xi for all layers
	 * @param Xi
	 * @return
	 */
	public int scoreGraph(int[][] Xi) {

		//first calculate the clause output for vertex layer (layer 1)
		calculate_clause_output(Xi[0], ta_state, clause_output, clause_vertex, output_one_vertices, PREDICT);

		for(int layer = 0; layer < ta_state_message.size(); layer++) {
			//calculate the clause output for message layer (layer 2)
			calculate_clause_output(Xi[layer+1], ta_state_message.get(layer),
					clause_output_message.get(layer),
					clause_vertex_message.get(layer),
					output_one_vertices_message.get(layer),
					PREDICT);
		}
		return sum_up_class_votes_graph();
	}

	public int[] getClause_output_unwrapped() {

		int[] clause_output_unwrapped = new int[nClauses];

		for (int i = 0; i < nClauses; i++) {
			int clause_chunk = i / INT_SIZE;
			int clause_pos = i % INT_SIZE;

			clause_output_unwrapped[i] = (clause_output[clause_chunk] & (1 << clause_pos)) != 0 ? 1 : 0;
		}

		return clause_output_unwrapped;

	}



	public int[] interpretablePrediction(int[] Xi, int max_class) {

		int j, k;
		int action_include, action_include_negated;

		int[] local_feature_strength = new int[2*nFeatures + 1];


		for (j = 0; j < nClauses; j++) {

			int clause_chunk = j / INT_SIZE;
			int clause_pos = j % INT_SIZE;

			if( ((clause_output[clause_chunk] & (1 << clause_pos)) != 0) && clause_weights[j] > 0) {

				for (k = 0; k < nFeatures; k++) {

					int la_chunk = k / INT_SIZE;
					int chunk_pos = k % INT_SIZE;

					action_include = tm_action(j, k);
					action_include_negated = tm_action(j, nFeatures + k);

					if(action_include == 1 && ((Xi[clause_vertex[j]*la_chunks + la_chunk] & (1 << chunk_pos)) != 0)) {
						local_feature_strength[k]++;
					}

					if(action_include_negated == 1 && ((Xi[clause_vertex[j]*la_chunks + la_chunk] & (1 << chunk_pos)) == 0) ) {
						local_feature_strength[nFeatures + k]++;
					}
				}
			}
		}

		local_feature_strength[local_feature_strength.length - 1] = max_class;
		return local_feature_strength;
	}


	/**
	 * Gives the strength of the features from the clauses
	 * given positive polarity, namely even
	 * indexed clauses j % 2 = 0
	 * Should only be used for multiclass classification
	 * and only on the targeted risky class (eg. failure, test positive, etc)
	 * otherwise, hard to interpret
	 * @param Xi
	 * @return
	 */
	public int[] riskPrediction(int[] Xi, boolean positive_polarity) {

		int j, k;
		int action_include, action_include_negated;

		int[] risk_strength = new int[2*nFeatures];
		int positive_pol = -1;

		if(!positive_polarity) positive_pol = 1;

		for (j = 0; j < nClauses; j++) {

			int clause_chunk = j / INT_SIZE;
			int clause_pos = j % INT_SIZE;

			if( ((clause_output[clause_chunk] & (1 << clause_pos)) != 0) && (clause_weights[j] * positive_pol < 0)) {

				for (k = 0; k < nFeatures; k++) {

					int la_chunk = k / INT_SIZE;
					int chunk_pos = k % INT_SIZE;

					action_include = tm_action(j, k);
					action_include_negated = tm_action(j, nFeatures + k);

					if(action_include == 1 && ((Xi[clause_vertex[j]*la_chunks + la_chunk] & (1 << chunk_pos)) != 0)) {
						risk_strength[k]++;
					}

					if(action_include_negated == 1 && ((Xi[clause_vertex[j]*la_chunks + la_chunk] & (1 << chunk_pos)) == 0) ) {
						risk_strength[nFeatures + k]++;
					}
				}
			}
		}
		return risk_strength;
	}











	/**
	 * Explainability interface goes here
	 *
	 * The idea is to have an overview of the accountability of each
	 * input feature that either supports a positive output from a clause
	 * or its negation supports a positive output of a clause
	 *
	 * If a feature is included in a clause more often than it is negated, and
	 * the collection of clauses has positive output, then the idea
	 * is that the highly prominent features were positive reinforcement in its
	 * decision
	 *
	 * @return int array of length 2*nFeatures where 0 < n < nFeatures are the positive
	 * identifying traits and the nFeatures < n < 2*nFeatures are the negative
	 */
	public int[] computeFeatureStrength() {

		int[] feature_strength = new int[2*nFeatures];

		for(int k = 0; k < nFeatures; k++) {
			for(int j = 0; j < nClauses; j++) {

				feature_strength[k] += tm_action(j, k)*clause_weights[j];
				feature_strength[2*k] += tm_action(j, nFeatures +k)*clause_weights[j];
			}
		}
		return feature_strength;
	}

	public int[] computeWeightedFeatureStrength() {

		int[] feature_strength = new int[2*nFeatures];

		for(int k = 0; k < nFeatures; k++) {
			for(int j = 0; j < nClauses; j++) {

				feature_strength[k] += tm_action(j, k)*clause_weights[j];
				feature_strength[2*k] += tm_action(j, nFeatures +k)*clause_weights[j];
			}
		}
		return feature_strength;
	}

	public ArrayList<Integer> getLiteralActionsInClause(int clause) {

		ArrayList<Integer> literal_actions = new ArrayList<Integer>();

		for(int k = 0; k < nFeatures; k++) {
			if(tm_action(clause, k) == 1) {
				literal_actions.add(k);
			}
			if(tm_action(clause, nFeatures + k) == 1) {
				literal_actions.add(nFeatures + k);
			}
		}

		return literal_actions;
	}

	/**
	 * Computes the feature strength per clause for organizing clause patterns
	 */
	public void computeWeightedFeatureStrengthByClause() {

		clause_feature_strength = new int[nClauses][2*nFeatures];

		for(int j = 0; j < nClauses; j++) {

			for(int k = 0; k < nFeatures; k++) {
				clause_feature_strength[j][k] += tm_action(j, k);
				clause_feature_strength[j][2*k] += tm_action(j, nFeatures + k);
			}
		}
	}



	/**
	 * Computes for a given position in time
	 * @param index from 0 < index < dim_y - patch_dim_y
	 * @return
	 */
//	public int[] computeConditionalFeatureStrength(int index) {
//
//		int dim_x = encoder.getDimX();
//		int dim_y = encoder.getDimY();
//		int dim_z = encoder.getDimZ();
//
//		int patch_dim_y = encoder.getDimPatchY();
//		int patch_dim_x = encoder.getDimPatchX();
//
//		index = index%(dim_y - patch_dim_y);
//
//		int[] index_patch = new int[patch_dim_y];
//		for(int j = 0; j < nClauses; j++) {
//
//			if(tm_action(j, index) == 1) {
//
//				for (int p_y = 0; p_y < patch_dim_y; ++p_y) {
//					for (int p_x = 0; p_x < patch_dim_x; ++p_x) {
//						for (int z = 0; z < dim_z; ++z) {
//
//							int patch_pos = (dim_y - patch_dim_y) + (dim_x - patch_dim_x) + p_y * patch_dim_x * dim_z + p_x * dim_z + z;
//							index_patch[p_y] += tm_action(j, patch_pos)*clause_weights[j];
//						}
//					}
//				}
//			}
//		}
//		return index_patch;
//	}


//	public ClauseDescriptor outputClause(int clause_number) {
//
//		int dim_x = encoder.getDimX();
//		int dim_y = encoder.getDimY();
//		int dim_z = encoder.getDimZ();
//
//		int patch_dim_y = encoder.getDimPatchY();
//		int patch_dim_x = encoder.getDimPatchX();
//
//		int max_threshold = 0;
//		int min_threshold = dim_y-1;
//
//		ClauseDescriptor clause = new ClauseDescriptor();
//
//		for(int index = 0; index < dim_y - patch_dim_y; index++) {
//
//			if(tm_action(clause_number, index) == 1 && max_threshold < index) {
//				max_threshold = index;
//			}
//			if(tm_action(clause_number, nFeatures + index) == 1 && min_threshold > index) {
//				min_threshold = index;
//			}
//		}
//		int clause_location = max_threshold;
//		clause.setInterval(max_threshold, min_threshold);
//
//
//		float[][] output = new float[2][patch_dim_y];
//
//		for (int p_y = 0; p_y < patch_dim_y; ++p_y) {
//
//			max_threshold = 0;
//			min_threshold = patch_dim_x-1;
//
//			for (int p_x = 0; p_x < patch_dim_x; ++p_x) {
//				for (int z = 0; z < dim_z; ++z) {
//
//					int patch_pos = (dim_y - patch_dim_y) + (dim_x - patch_dim_x) + p_y * patch_dim_x * dim_z + p_x * dim_z + z;
//
//					if(tm_action(clause_number, patch_pos) == 1 && max_threshold < p_x) {
//						max_threshold = p_x;
//					}
//					if(tm_action(clause_number, nFeatures + patch_pos) == 1 && min_threshold > p_x) {
//						min_threshold = p_x;
//					}
//
//				}
//			}
//		}
//		clause.setFeature_strength(output);
//
//		return clause;
//	}

	/**
	 * Returns the most n relevant clauses based on clause weighting
	 * We sort the clause weights, and return the top n clauses that
	 * are contributing to the positive output.
	 *
	 * Only used if this particular Tsetlin when the output is largest
	 * output of all classes

	 * @return
	 */
//	public ClauseDescriptor[] getTopPatterns(int n) {
//
//		clause_index = new int[clause_weights.length];
//        for(int i = 0; i < clause_index.length; i++) {
//        	clause_index[i] = i;
//        }
//
//		int[] copy_clause_weights = ArrayUtils.clone(clause_weights);
//		QuickSort.sort(copy_clause_weights, clause_index);
//        ArrayUtils.reverse(copy_clause_weights);
//        ArrayUtils.reverse(clause_index);
//
//        ClauseDescriptor[] top_clauses = new ClauseDescriptor[n];
//
//        for(int i = 0; i < n; i++) {
//        	if(clause_index[i]%2 == 0) {
//        		top_clauses[i] = outputClause(clause_index[i]);
//        	}
//        }
//
//        return top_clauses;
//	}

	public int[] getSortedClauses() {

		clause_index = new int[clause_weights.length];
		for(int i = 0; i < clause_index.length; i++) {
			clause_index[i] = i;
		}

		int[] copy_clause_weights = ArrayUtils.clone(clause_weights);
		QuickSort.sort(copy_clause_weights, clause_index);
		ArrayUtils.reverse(copy_clause_weights);
		ArrayUtils.reverse(clause_index);

		return clause_index;
	}



	public int[] getGlobalFeatureImportance() {
		return computeWeightedFeatureStrength();
	}

	/**
	 * Returns the weights of the clauses for this class
	 *
	 * @return
	 */
	public int[] getClauseStrength() {
		return clause_weights;
	}

	public int getClauseOutput(int j) {
		return clause_output[j];
	}

	public int[] getClauseOutput() {
		return clause_output;
	}

	public int getNumberOfPatches() {
		return number_of_vertices;
	}

	public int getNumberOfTAChunks() {
		return la_chunks;
	}


	public void setMaxLearnRate(float learn_rate) {
		this.max_specificity = learn_rate;

	}

	public void setY_min(float target_min) {

	}

	public void setY_max(float target_max) {
	}

	public void reset_clause_output() {
		for(int i = 0; i < clause_output.length; i++) {
			clause_output[i] = ~0;
		}
	}


	public int getT() {
		return T;
	}

	public static void main(String[] args) throws Exception {



	}

	public int getThreshold() {
		return T;
	}

	public int[] getFeedback() {

		return feedback_to_clauses;
	}

	public int getSTATE_BITS() {
		// TODO Auto-generated method stub
		return STATE_BITS;
	}

	public int getNumberOfFeatures() {
		return nFeatures;
	}

	public int getAction(int clause_number, int index) {
		// TODO Auto-generated method stub
		return 0;
	}

	public double getClass_probability() {
		return class_probability;
	}

	public void setClass_probability(double class_probability) {
		this.class_probability = class_probability;
	}


	public float[] getClause_patch_coverage() {
		return clause_vertex_coverage;
	}

	public int getnClauses() {
		return nClauses;
	}


	public int getMax_number_literals() {
		return max_number_literals;
	}

	public void setMax_number_literals(int max_number_literals) {
		this.max_number_literals = max_number_literals;
	}
	public int[][] getClause_feature_strength() {
		return clause_feature_strength;
	}

	public void setClause_feature_strength(int[][] clause_feature_strength) {
		this.clause_feature_strength = clause_feature_strength;
	}

	public void setMaxNumberOfLiterals(int max_literals) {
		setMax_number_literals(max_literals);
	}

	public int[] getClause_weights() {
		return clause_weights;
	}

	public int[][][] getState() {
		return ta_state;
	}

	/**
	 * Composition of clause output
	 * 	 Only works for 2 layers at the moment
	 * 	 TODO: Generalize to n layers
	 * @param clause_chunk int chunk of the clause
	 * @return outout 1 or 0
	 */
	public int compositeClauseOutput(int clause_chunk) {
		return !clause_output_message.isEmpty() ? clause_output[clause_chunk] & clause_output_message.get(0)[clause_chunk] : clause_output[clause_chunk];
	}

	public final int[][][] getTa_state() {
		return ta_state;
	}



















}
