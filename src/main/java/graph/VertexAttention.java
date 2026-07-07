package graph;

import encoding.hypervector.SparseBinaryVector;

import java.util.ArrayList;
import java.util.List;

public class VertexAttention {

    private final List<SparseBinaryVector> layerVectors = new ArrayList<>();

    /**
     * Adds a layer vector to the vertex.
     * @param layerVector The layer vector to add.
     */
    public void addLayerVector(SparseBinaryVector layerVector) {
        this.layerVectors.add(layerVector);
    }

    public void setLayerVector(int index, SparseBinaryVector layerVector) {
        if (index < 0 || index >= layerVectors.size()) {
            throw new IndexOutOfBoundsException("Index out of bounds for layer vector");
        }
        this.layerVectors.set(index, layerVector);
    }

    /**
     * Returns the layer vector for the specified layer.
     * layer 0 is the vertex data, layer 1 is the vertex bind message data, and so on.
     * @param layer
     * @return
     */
    public SparseBinaryVector getLayerVector(int layer) {
        if (layer < 0 || layer >= layerVectors.size()) {
            throw new IndexOutOfBoundsException("Index out of bounds for layer vector");
        }
        return layerVectors.get(layer);
    }

    public List<SparseBinaryVector> getLayerVectors() {
        return layerVectors;
    }
}
