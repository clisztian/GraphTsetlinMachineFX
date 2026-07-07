package graph;

import java.util.ArrayList;
import java.util.List;

public class GraphSample {
    public final int graphId;
    public final int label;
    private int time;



    public final List<TsetlinVertex> vertices = new ArrayList<>();
    public final List<TsetlinEdgeRecord> edges = new ArrayList<>();

    public GraphSample(int graphId, int label) {
        this.graphId = graphId;
        this.label = label;
    }

    public int getLabel() {
        return label;
    }

    public record TsetlinEdgeRecord(int sourceIndex, int targetIndex, TsetlinEdge edge) {}

    public int getTime() {
        return time;
    }

    public void setTime(int time) {
        this.time = time;
    }
}

