package graph;

import encoding.hypervector.SparseBinaryVector;
import graphview.SmartLabelSource;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a vertex in a Tsetlin graph.
 * The vertex can represent a feature, a node, or any other entity in the graph.
 */
public class TsetlinVertex {

    private int vertexId;
    // Unique identifier for the vertex, can be set externally or generated internally
    private String vertexName;
    private float vertexImportance;
    private final List<VertexAttention> vertexData = new ArrayList<>();

    /**
     * Instantiates a new Tsetlin vertex.
     * @param vertexName A name for the vertex, for example the feature name
     * @param vertexImportance Importance of vertex, for example  feature importance, or PageRank,or other metric
     */
    public TsetlinVertex(String vertexName, float vertexImportance) {
        this.vertexName = vertexName;
        this.vertexImportance = vertexImportance;
    }

    public void initializeLayerData(SparseBinaryVector data) {

        VertexAttention vertexAttention = new VertexAttention();
        vertexAttention.addLayerVector(data);

        vertexData.add(vertexAttention);
    }

    /**
     * Add vertex data to the list.
     * @param data
     */
    public void addLayerVertexData(int index, SparseBinaryVector data) {
        VertexAttention v = getVertexData(index);
        v.addLayerVector(data);
    }

    /**
     * Get the vertex data at the specified index.
     * @param index
     * @return
     */
    public final VertexAttention getVertexData(int index) {
        if (index < 0 || index >= vertexData.size()) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + vertexData.size());
        }
        return vertexData.get(index);
    }

    public String getVertexName() {
        return vertexName;
    }

    /**
     * Set the vertex name.
     * @param vertexName
     */
    public void setVertexName(String vertexName) {
        this.vertexName = vertexName;
    }

    /**
     * Get the vertex importance.
     * @return
     */
    public float getVertexImportance() {
        return vertexImportance;
    }

    /**
     * Set the vertex importance.
     * @param vertexImportance
     */
    public void setVertexImportance(float vertexImportance) {
        this.vertexImportance = vertexImportance;
    }

    /**
     * Get the vertex data.
     * @return
     */
    public List<VertexAttention> getVertexData() {
        return vertexData;
    }


    @SmartLabelSource
    public String getName() {
        return vertexName + " " + vertexData.size();
    }

    @Override
    public String toString() {
        return "TsetlinVertex{" +
                "vertexName='" + vertexName + '\'' +
                ", vertexImportance=" + vertexImportance +
                ", numberSamples=" + vertexData.size() +
                '}';
    }

    public int getVertexId() {
        return vertexId;
    }

    public void setVertexId(int vertexId) {
        this.vertexId = vertexId;
    }
}
