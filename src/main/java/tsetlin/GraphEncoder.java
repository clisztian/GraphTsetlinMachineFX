package tsetlin;

public class GraphEncoder {

    private final int featureDimension;
    private final int messageDimension;
    private final int numberLayers;



    private final int totalNumberFeatures;
    private final int dim_x;

    private final int number_of_vertices; //total vertices used for the graph
    private final int number_of_ta_chunks;
    private final int layer_ta_chunks;


    private final int append_negated = 1;
    private final int featureDimensionPerLayer;

    /**
     * For classic Tsetlin Machine, just takes the entire feature space
     * @param dim_x
     */
    public GraphEncoder(int dim_x) {
        this.number_of_vertices = 1;
        this.dim_x = dim_x;
        this.totalNumberFeatures = dim_x;
        this.featureDimension = dim_x;
        this.featureDimensionPerLayer = featureDimension * number_of_vertices;
        this.number_of_ta_chunks = (((2* totalNumberFeatures -1)/32 + 1));
        this.layer_ta_chunks = number_of_ta_chunks;

        this.messageDimension = 0;
        this.numberLayers = 1;
    }

    /**
     * Constructor for GraphEncoder
     * Message space is the projection of the feature space with the edge space, and should be equal to the feature space
     * @param number_of_vertices
     * @param feature_dim
     * @param message_dim
     * @param number_layers
     */
    public GraphEncoder(int number_of_vertices, int feature_dim, int message_dim, int number_layers) {

        //if the feature_dim and message_dim are not equal or multiples of 32, throw an error
        if (feature_dim % 32 != 0 || message_dim % 32 != 0) {
            throw new IllegalArgumentException("feature_dim and message_dim must be multiples of 32");
        }

        //if message_dim is not equal to feature_dim, throw an error
        if (feature_dim != message_dim) {
            throw new IllegalArgumentException("feature_dim and message_dim must be equal");
        }


        this.featureDimension = feature_dim;
        this.messageDimension = message_dim;
        this.numberLayers = number_layers;
        this.number_of_vertices = number_of_vertices;
        this.totalNumberFeatures = number_of_vertices *(feature_dim + (number_layers-1)*message_dim);
        this.featureDimensionPerLayer = feature_dim * number_of_vertices;

        this.number_of_ta_chunks = (((2* feature_dim -1)/32 + 1));

        this.layer_ta_chunks = (((2* feature_dim -1)/32 + 1));

        this.dim_x = feature_dim + number_layers*message_dim;
    }





    public int getNumber_of_vertices() {
        return number_of_vertices;
    }



    public int getNumber_of_ta_chunks() {
        return number_of_ta_chunks;
    }



    public int getTotalNumberFeatures() {
        return totalNumberFeatures;
    }


    public int getDim_x() {
        return dim_x;
    }

    public int getFeatureDimension() {
        return featureDimension;
    }

    public int getMessageDimension() {
        return messageDimension;
    }

    public int getNumberLayers() {
        return numberLayers;
    }

    public int getLayer_ta_chunks() {
        return layer_ta_chunks;
    }









    /**
     * Packs a flat bit-array into ints, 32 bits per int.
     *
     * @param bits               the input array of 0/1 values, length = numberVertices * numberFeatureBits
     * @param numberVertices     how many “vertices” (rows) you have
     * @param numberFeatureBits  how many bits per vertex (must be a multiple of 32)
     * @return                   an int[] of length numberVertices * numberFeatureChunks
     */
    public static int[] packBitsToInts(int[] bits, int numberVertices, int numberFeatureBits) {
        if (numberFeatureBits % 32 != 0) {
            throw new IllegalArgumentException("numberFeatureBits must be a multiple of 32");
        }
        int numberFeatureChunks = numberFeatureBits / 32;
        int[] packed = new int[numberVertices * numberFeatureChunks];

        for (int v = 0; v < numberVertices; v++) {
            int vertexOffsetBits = v * numberFeatureBits;
            int vertexOffsetInts = v * numberFeatureChunks;

            for (int chunk = 0; chunk < numberFeatureChunks; chunk++) {
                int value = 0;
                int chunkOffsetBits = vertexOffsetBits + chunk * 32;
                for (int b = 0; b < 32; b++) {
                    // shift left to make room, then OR in next bit
                    value = (value << 1) | (bits[chunkOffsetBits + b] & 1);
                }
                packed[vertexOffsetInts + chunk] = value;
            }
        }
        return packed;
    }


}
