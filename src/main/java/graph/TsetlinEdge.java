package graph;

import encoding.hypervector.SparseBinaryVector;
import graphview.SmartLabelSource;
/**
 * Represents an edge in a Tsetlin graph.
 */
public class TsetlinEdge {

    private final String edgeName;
    private final float edgeImportance;
    private final SparseBinaryVector edgeEncoder;

    /**
     * Instantiates a new Tsetlin edge.
     * @param edgeName A name for the edge, for example the correlation name
     * @param edgeImportance Importance of edge, for example the correlation strength
     * @param edgeEncoder The encoder for the edge, for example a directional encoder, or a cost/distance encoder
     */
    public TsetlinEdge(String edgeName, float edgeImportance, SparseBinaryVector edgeEncoder) {
        this.edgeName = edgeName;
        this.edgeImportance = edgeImportance;
        this.edgeEncoder = edgeEncoder;
    }

    public String getEdgeName() {
        return edgeName;
    }

    public float getEdgeImportance() {
        return edgeImportance;
    }

    public SparseBinaryVector getEdgeEncoder() {
        return edgeEncoder;
    }

    @SmartLabelSource
    public String getDisplay() {
        /* If the above annotation is not present, the toString()
        will be used as the edge label. */

        return edgeName;
    }

    @Override
    public String toString() {
        return edgeName;
    }

}
