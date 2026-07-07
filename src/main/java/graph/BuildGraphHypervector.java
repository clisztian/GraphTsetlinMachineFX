package graph;

import encoding.hypervector.SparseBinaryVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

public class BuildGraphHypervector {
    private static final Logger logger = LoggerFactory.getLogger(BuildGraphHypervector.class);

    /**
     * Builds a hypervector for a given index in the TsetlinAttentionGraph
     *
     * Assumes that we have both node and edge labels for the embeddings which are used
     * to construct the hypervector.
     *
     *
     * @param graph
     * @param index
     * @return
     */
    public static SparseBinaryVector buildHypervectorEdgeNodeLabels(TsetlinAttentionGraph graph, int index) {

        //first ensure that data exists for the given index, if graph.getVertexKeys().get(0).getVertexData().get(index) == null throw an exception
        if (graph.getVertexKeys().get(0).getVertexData(index) == null) {
            throw new IllegalArgumentException("No data available for index: " + index);
        }

        List<SparseBinaryVector> bundledAttention = new ArrayList<>();

        Map<SparseBinaryVector, TsetlinEdge> edgeMap = new HashMap<>();

        //loop through all vertices and find all the edges for the vertex at index
        for (TsetlinVertex vertex : graph.getVertexKeys()) {

            //get the hypervector for the vertex at index
            SparseBinaryVector vertexHypervector = vertex.getVertexData(index).getLayerVector(0);

            //log the first segment of the vertex hypervector
//            logger.info("Vertex {} hypervector segment: {}",
//                    vertex.getVertexName(),vertexHypervector.getSegments()[0]);

            //order of edges doesn't matter here, get all incoming messages
            Collection<Edge<TsetlinEdge,TsetlinVertex>> edges = graph.outboundEdges(graph.vertices.get(vertex));

            //with incoming edges, get the vertex data and the edge data
            for (Edge<TsetlinEdge,TsetlinVertex> edge : edges) {


                TsetlinEdge tsetlinEdge = edge.element();
                SparseBinaryVector bindedEdge = vertexHypervector.bind(tsetlinEdge.getEdgeEncoder());
                SparseBinaryVector outgoingVector = edge.vertices()[1].element().getVertexData().get(index).getLayerVector(0);

                //SparseBinaryVector outgoingVertex = outgoingVector.bind(graph.getEmbedding(edge.vertices()[1].element()));

//                logger.info("Outgoing vertex {} hypervector segment: {}",
//                        edge.vertices()[1].element().getVertexName(), outgoingVertex.getSegments()[0]);

                SparseBinaryVector encodedEdge = bindedEdge.bind(outgoingVector);

//                logger.info("Encoded edge hypervector segment: {}",
//                        encodedEdge.getSegments()[0]);

                SparseBinaryVector encodedVertex = encodedEdge.bind(graph.getEmbedding(vertex));

//                logger.info("Encoded vertex {} ",
//                        String.format("%64s", Long.toBinaryString(encodedVertex.getSegments()[0])).replace(' ', '0'));

                edgeMap.put(encodedVertex, tsetlinEdge);

                bundledAttention.add(encodedVertex);
            }
        }

        //set the edge map in the graph
        graph.setEdgeMap(edgeMap);

        return SparseBinaryVector.bundleWeighted(bundledAttention);
    }

    public static SparseBinaryVector buildHypervectorParallel(TsetlinAttentionGraph graph, int index) {
        if (graph.getVertexKeys().get(0).getVertexData(index) == null) {
            throw new IllegalArgumentException("No data available for index: " + index);
        }

        // Each vertex's contribution is computed independently
        List<SparseBinaryVector> partialBundles = graph.getVertexKeys()
                .parallelStream()
                .map(vertex -> {
                    SparseBinaryVector vertexHypervector = vertex.getVertexData(index).getLayerVector(0);
                    List<Edge<TsetlinEdge, TsetlinVertex>> edges =
                            List.copyOf(graph.incidentEdges(graph.vertices.get(vertex)));

                    List<SparseBinaryVector> edgeEncodings = new ArrayList<>();

                    for (Edge<TsetlinEdge, TsetlinVertex> edge : edges) {
                        if (edge.vertices()[1].element() != vertex) continue; // Only outbound

                        TsetlinEdge tsetlinEdge = edge.element();
                        SparseBinaryVector edgeHypervector = graph.getEmbedding(tsetlinEdge).bind(tsetlinEdge.getEdgeEncoder());
                        SparseBinaryVector bindedEdge = vertexHypervector.bind(edgeHypervector);

                        TsetlinVertex childVertex = edge.vertices()[0].element();
                        SparseBinaryVector childEncoding = childVertex.getVertexData().get(index).getLayerVector(0)
                                .bind(graph.getEmbedding(childVertex));

                        SparseBinaryVector message = bindedEdge.bind(childEncoding);
                        SparseBinaryVector routedMessage = message.bind(graph.getEmbedding(vertex));

                        edgeEncodings.add(routedMessage);
                    }

                    // Combine this vertex's edge encodings into a single contribution
                    return edgeEncodings.isEmpty() ?
                            SparseBinaryVector.getZeroVector() :
                            SparseBinaryVector.bundle(edgeEncodings);
                })
                .collect(Collectors.toList());

        return SparseBinaryVector.bundle(partialBundles);
    }


    public static SparseBinaryVector buildHypervectorThreadPool(TsetlinAttentionGraph graph, int index, int numThreads) {
        if (graph.getVertexKeys().get(0).getVertexData(index) == null) {
            throw new IllegalArgumentException("No data available for index: " + index);
        }

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<SparseBinaryVector>> futures = new ArrayList<>();

        for (TsetlinVertex vertex : graph.getVertexKeys()) {
            futures.add(executor.submit(() -> {
                SparseBinaryVector vertexHypervector = vertex.getVertexData(index).getLayerVector(0);
                List<Edge<TsetlinEdge, TsetlinVertex>> edges = List.copyOf(graph.incidentEdges(graph.vertices.get(vertex)));

                List<SparseBinaryVector> edgeEncodings = new ArrayList<>();

                for (Edge<TsetlinEdge, TsetlinVertex> edge : edges) {
                    if (edge.vertices()[1].element() != vertex) continue;

                    TsetlinEdge tsetlinEdge = edge.element();
                    SparseBinaryVector edgeHypervector = graph.getEmbedding(tsetlinEdge).bind(tsetlinEdge.getEdgeEncoder());
                    SparseBinaryVector bindedEdge = vertexHypervector.bind(edgeHypervector);

                    TsetlinVertex childVertex = edge.vertices()[0].element();
                    SparseBinaryVector childEncoding = childVertex.getVertexData().get(index).getLayerVector(0)
                            .bind(graph.getEmbedding(childVertex));

                    SparseBinaryVector message = bindedEdge.bind(childEncoding);
                    SparseBinaryVector routedMessage = message.bind(graph.getEmbedding(vertex));

                    edgeEncodings.add(routedMessage);
                }

                return edgeEncodings.isEmpty()
                        ? SparseBinaryVector.getZeroVector()
                        : SparseBinaryVector.bundle(edgeEncodings);
            }));
        }

        executor.shutdown();

        List<SparseBinaryVector> bundledVectors = new ArrayList<>();
        for (Future<SparseBinaryVector> future : futures) {
            try {
                bundledVectors.add(future.get());
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException("Thread execution failed", e);
            }
        }

        return SparseBinaryVector.bundle(bundledVectors);
    }

}
