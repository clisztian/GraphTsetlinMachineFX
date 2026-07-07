package util;


import encoding.embedding.LinearEmbedding;
import encoding.hypervector.SparseBinaryVector;
import graph.GraphSample;
import graph.TsetlinEdge;
import graph.TsetlinVertex;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.util.*;

//SparseBinaryVector encoding = nodeEncodings.getOrDefault(nodeId, SparseBinaryVector.randVector());

public class TUDatasetLoader {

    static Random random = new Random(12);
    static DecimalFormat decimalFormat = new DecimalFormat("#.#####");

    /**
     * If we have no edge labels, we can use a default edge encoder.
     */
    private static final SparseBinaryVector rightEdgeEncoder = SparseBinaryVector.randVector();
    private static final SparseBinaryVector leftEdgeEncoder = SparseBinaryVector.randVector();

    private static final SparseBinaryVector[] roleVectors = new SparseBinaryVector[5];
    //initialize the role vectors with random vectors
    static {
        for (int i = 0; i < roleVectors.length; i++) {
            roleVectors[i] = SparseBinaryVector.randVector();
        }
    }

    //create a mapping for the labels of the nodes and edges
    /*
    Node labels:

  0  C
  1  N
  2  O
  3  F
  4  I
  5  Cl
  6  Br

Edge labels:

  0  aromatic
  1  single
  2  double
  3  triple
     */


    private static Map<Integer, String> nodeLabelMap = Map.of(
            0, "C",
            1, "N",
            2, "O",
            3, "F",
            4, "I",
            5, "Cl",
            6, "Br"
    );

    private static Map<Integer, String> edgeLabelMap = Map.of(
            0, "aromatic",
            1, "single",
            2, "double",
            3, "triple"
    );


    public static List<GraphSample> loadGraphs(Path datasetDir, String datasetName) throws IOException {

        String prefix = datasetName + "_";

        //maps class labels such as -1, 0, 1 to integers 0,1,2...
        Map<Integer, Integer> classLabels = new TreeMap<>();



        Map<Integer, SparseBinaryVector> nodeEncodings = new HashMap<>();
        Map<Integer, SparseBinaryVector> edgeEncodings = new HashMap<>();
// Load core files
        Map<Integer, Integer> nodeToGraph = loadGraphIndicators(datasetDir.resolve(prefix + "graph_indicator.txt"));
        Map<Integer, Integer> graphLabels = loadGraphLabels(datasetDir.resolve(prefix + "graph_labels.txt"));
        Map<Integer, Integer> nodeLabels = tryLoadNodeLabels(datasetDir.resolve(prefix + "node_labels.txt"));
        List<int[]> edgeList = loadEdges(datasetDir.resolve(prefix + "A.txt"));

        // Optional: edge labels and attributes
        List<String> edgeLabelLines = tryReadLines(datasetDir.resolve(prefix + "edge_labels.txt"));
        List<String> edgeAttrLines = tryReadLines(datasetDir.resolve(prefix + "edge_attributes.txt"));
        //node attributes
        Map<Integer, double[]> nodeAttributes = tryLoadNodeAttributes(datasetDir.resolve(prefix + "node_attributes.txt"));

        boolean hasEdgeLabels = edgeLabelLines != null && edgeLabelLines.size() == edgeList.size();
        boolean hasEdgeAttrs = edgeAttrLines != null && edgeAttrLines.size() == edgeList.size();
        boolean hasNodeAttrs = nodeAttributes != null && nodeAttributes.size() == nodeToGraph.size();

        LinearEmbedding nodeEmbedding = new LinearEmbedding(0, 1, 10);

        //map graph labels to integers, and ensure they are in sequential order by key so that -2,-1,0,1,2 would map to 0,1,2,3,4...
        //first we get the unique labels, then we sort them and map them to integers
        Set<Integer> uniqueLabels = new HashSet<>(graphLabels.values());
        List<Integer> sortedLabels = new ArrayList<>(uniqueLabels);
        Collections.sort(sortedLabels);
        for (int i = 0; i < sortedLabels.size(); i++) {
            classLabels.put(sortedLabels.get(i), i);
        }


        // Group nodes by graph
        Map<Integer, List<Integer>> graphToNodeIds = new HashMap<>();
        nodeToGraph.forEach((nodeId, graphId) ->
                graphToNodeIds.computeIfAbsent(graphId, k -> new ArrayList<>()).add(nodeId)
        );

        // Build GraphSample objects
        Map<Integer, GraphSample> graphs = new HashMap<>();
        Map<Integer, Integer> nodeGlobalToLocalIndex = new HashMap<>();
        for (Map.Entry<Integer, List<Integer>> entry : graphToNodeIds.entrySet()) {
            int graphId = entry.getKey();
            int label = classLabels.getOrDefault(graphLabels.get(graphId), 0); // Default to 0 if not found
            GraphSample graph = new GraphSample(graphId, label);

            List<Integer> nodeIds = entry.getValue();
            for (int i = 0; i < nodeIds.size(); i++) {
                int nodeId = nodeIds.get(i);

                //get label of node from the mapping
                String nodeName = "v";

                if(datasetName.equals("MUTAG")) {
                    nodeName = nodeLabelMap.getOrDefault(nodeLabels.get(nodeId), "Unknown");
                }

                String name = nodeName + " " + nodeId;

                List<SparseBinaryVector> nodeBundledAttributes = new ArrayList<>();
                //add zero vector if no attributes are available

                if (nodeAttributes != null && nodeAttributes.containsKey(nodeId)) {
                    double[] attributes = nodeAttributes.get(nodeId);

                    //create a bundled SparseBinaryVector from the attributes using the LinearEmbedding which are normalized between 0 and 1
                    for (int k = 0; k < attributes.length; k++) {
                        SparseBinaryVector attrVector = nodeEmbedding.forward(attributes[k]).bind(roleVectors[k % roleVectors.length]);
                        nodeBundledAttributes.add(attrVector);
                    }
                }
                else {

                    nodeBundledAttributes.add(SparseBinaryVector.getZeroVector());
                }


                SparseBinaryVector nodeSymbolEncoding = nodeEncodings.getOrDefault(nodeId, SparseBinaryVector.randVector());
                SparseBinaryVector bundledAttributes = SparseBinaryVector.bundleWeighted(nodeBundledAttributes);


                TsetlinVertex vertex = new TsetlinVertex(name, i);
                vertex.initializeLayerData(bundledAttributes.bind(nodeSymbolEncoding));
                graph.vertices.add(vertex);

                nodeGlobalToLocalIndex.put(nodeId, i); // global node ID → local index in this graph
            }

            graphs.put(graphId, graph);
        }

        // Add edges
        int edgeIdx = 0;
        for (int[] edge : edgeList) {
            int src = edge[0];
            int dst = edge[1];
            int g1 = nodeToGraph.get(src);
            int g2 = nodeToGraph.get(dst);
            if (g1 != g2) continue; // skip inter-graph edges

            GraphSample graph = graphs.get(g1);
            int localSrc = nodeGlobalToLocalIndex.get(src);
            int localDst = nodeGlobalToLocalIndex.get(dst);

            // Edge importance from attribute
            float importance = 0.01f*random.nextFloat(); // Default random importance
            if (hasEdgeAttrs) {
                importance = Float.parseFloat(edgeAttrLines.get(edgeIdx).trim()) + importance; // Add random importance
            }

            // Edge encoder from label
            SparseBinaryVector edgeEncoder;
            if (hasEdgeLabels) {
                int label = Integer.parseInt(edgeLabelLines.get(edgeIdx).trim());
                edgeEncoder = edgeEncodings.getOrDefault(label, SparseBinaryVector.randVector());
            }
            else {
                //if src < dest, use right edge encoder, otherwise use left edge encoder
                edgeEncoder = (src < dst) ? rightEdgeEncoder : leftEdgeEncoder;
            }

            String edgeLabel = edgeIdx + " " + decimalFormat.format(importance);

            if(datasetName.equals("MUTAG")) {
                //if MUTAG, use the edge label map to get the edge name
                edgeLabel = edgeLabelMap.getOrDefault(Integer.parseInt(edgeLabelLines.get(edgeIdx).trim()), "Unknown") + " " + decimalFormat.format(importance);;

            }

            graph.edges.add(new GraphSample.TsetlinEdgeRecord(
                    localSrc, localDst,
                    new TsetlinEdge(edgeLabel, importance, edgeEncoder)
            ));
            edgeIdx++;
        }

        return new ArrayList<>(graphs.values());
    }


    public static List<GraphSample> loadTemporalGraphs(Path datasetDir, String datasetName) throws IOException {

        String prefix = datasetName + "_";

        Map<Integer, Integer> nodeToGraph = loadGraphIndicators(datasetDir.resolve(prefix + "graph_indicator.txt"));
        Map<Integer, Integer> graphLabels = loadGraphLabels(datasetDir.resolve(prefix + "graph_labels.txt"));
        List<int[]> edgeList = loadEdges(datasetDir.resolve(prefix + "A.txt"));
        List<String> edgeAttrLines = Files.readAllLines(datasetDir.resolve(prefix + "edge_attributes.txt"));
        List<String> nodeLabelLines = Files.readAllLines(datasetDir.resolve(prefix + "node_labels.txt"));

        Map<Integer, List<Integer>> graphToNodeIds = new HashMap<>();
        nodeToGraph.forEach((nodeId, graphId) ->
                graphToNodeIds.computeIfAbsent(graphId, k -> new ArrayList<>()).add(nodeId));

        Map<Integer, List<int[]>> nodeLabelTimelines = new HashMap<>();
        for (int nodeId = 1; nodeId <= nodeLabelLines.size(); nodeId++) {
            String[] tokens = nodeLabelLines.get(nodeId - 1).trim().split(",");
            List<int[]> timeline = new ArrayList<>();
            for (int i = 0; i < tokens.length - 1; i += 2) {
                int time = Integer.parseInt(tokens[i].trim());
                int label = Integer.parseInt(tokens[i + 1].trim());
                timeline.add(new int[]{time, label});
            }
            nodeLabelTimelines.put(nodeId, timeline);
        }

        Map<Integer, Integer> edgeAvailableTime = new HashMap<>();
        for (int i = 0; i < edgeAttrLines.size(); i++) {
            edgeAvailableTime.put(i, Integer.parseInt(edgeAttrLines.get(i).trim()));
        }

        int maxTime = edgeAvailableTime.values().stream().max(Integer::compareTo).orElse(0);

        List<GraphSample> snapshots = new ArrayList<>();

        LinearEmbedding nodeEmbedding = new LinearEmbedding(0, 1, 10);
        Map<Integer, SparseBinaryVector> nodeEncodings = new HashMap<>();

        System.out.println("Max time: " + maxTime);

        for (int t = 0; t <= maxTime; t++) {

            Map<Integer, GraphSample> graphMap = new HashMap<>();
            Map<Integer, Integer> nodeGlobalToLocal = new HashMap<>();
            Map<Integer, TsetlinVertex> nodeIdToVertex = new HashMap<>();
            int nodeLocalId = 0;

            for (Map.Entry<Integer, List<Integer>> entry : graphToNodeIds.entrySet()) {
                int graphId = entry.getKey();
                int label = graphLabels.getOrDefault(graphId, 0);
                GraphSample graph = new GraphSample(graphId, label);
                graph.setTime(t);

                for (int nodeId : entry.getValue()) {
                    List<int[]> timeline = nodeLabelTimelines.get(nodeId);
                    if (timeline == null || timeline.isEmpty()) continue;
                    int assignedLabel = -1;
                    for (int[] event : timeline) {
                        if (event[0] <= t) assignedLabel = event[1];
                        else break;
                    }
                    if (assignedLabel == -1) continue;

                    String name = "v" + nodeId;
                    SparseBinaryVector labelVector = nodeEmbedding.forward((double) assignedLabel / 10.0)
                            .bind(roleVectors[0]);
                    SparseBinaryVector nodeSymbol = nodeEncodings.computeIfAbsent(nodeId, k -> SparseBinaryVector.randVector());
                    SparseBinaryVector bundled = labelVector.bind(nodeSymbol);

                    TsetlinVertex vertex = new TsetlinVertex(name, nodeLocalId);
                    vertex.setVertexId(nodeId);
                    vertex.initializeLayerData(bundled);

                    graph.vertices.add(vertex);
                    nodeGlobalToLocal.put(nodeId, nodeLocalId);
                    nodeIdToVertex.put(nodeId, vertex);
                    nodeLocalId++;
                }

                graphMap.put(graphId, graph);
            }

            int edgeIdx = 0;
            for (int[] edge : edgeList) {
                int src = edge[0];
                int dst = edge[1];
                int availableTime = edgeAvailableTime.getOrDefault(edgeIdx, Integer.MAX_VALUE);
                if (availableTime > t) {
                    edgeIdx++;
                    continue;
                }
                int g1 = nodeToGraph.get(src);
                int g2 = nodeToGraph.get(dst);
                if (g1 != g2) {
                    edgeIdx++;
                    continue;
                }
                GraphSample graph = graphMap.get(g1);
                if (graph == null) {
                    edgeIdx++;
                    continue;
                }
                if (!nodeGlobalToLocal.containsKey(src) || !nodeGlobalToLocal.containsKey(dst)) {
                    edgeIdx++;
                    continue;
                }
                int localSrc = nodeGlobalToLocal.get(src);
                int localDst = nodeGlobalToLocal.get(dst);
                float importance = 0.01f * random.nextFloat();
                SparseBinaryVector edgeEncoder = (src < dst) ? rightEdgeEncoder : leftEdgeEncoder;
                String edgeLabel = edgeIdx + " t=" + t;
                graph.edges.add(new GraphSample.TsetlinEdgeRecord(
                        localSrc, localDst,
                        new TsetlinEdge(edgeLabel, importance, edgeEncoder)
                ));
                edgeIdx++;
                //System.out.println("edge count: " + graph.edges.size());
            }


            snapshots.addAll(graphMap.values());
        }

        return snapshots;
    }





//
//    public static List<GraphSample> loadTemporalGraphs(Path datasetDir, String datasetName) throws IOException {
//        String prefix = datasetName + "_";
//
//        Map<Integer, Integer> nodeToGraph = loadGraphIndicators(datasetDir.resolve(prefix + "graph_indicator.txt"));
//        Map<Integer, Integer> graphLabels = loadGraphLabels(datasetDir.resolve(prefix + "graph_labels.txt"));
//        List<int[]> edgeList = loadEdges(datasetDir.resolve(prefix + "A.txt"));
//        List<String> edgeAttrLines = Files.readAllLines(datasetDir.resolve(prefix + "edge_attributes.txt"));
//        List<String> nodeLabelLines = Files.readAllLines(datasetDir.resolve(prefix + "node_labels.txt"));
//
//        // Parse edge availability times
//        Map<Integer, Integer> edgeAvailableAt = new HashMap<>();
//        for (int i = 0; i < edgeAttrLines.size(); i++) {
//            edgeAvailableAt.put(i, Integer.parseInt(edgeAttrLines.get(i).trim()));
//        }
//
//        // Parse node label change timelines
//        Map<Integer, TreeMap<Integer, Integer>> nodeTimeline = new HashMap<>();
//        int maxTime = 0;
//        for (int i = 0; i < nodeLabelLines.size(); i++) {
//            String[] tokens = nodeLabelLines.get(i).trim().split(",");
//            TreeMap<Integer, Integer> timeline = new TreeMap<>();
//            for (int j = 0; j < tokens.length; j += 2) {
//                int t = Integer.parseInt(tokens[j].trim());
//                int label = Integer.parseInt(tokens[j + 1].trim());
//                timeline.put(t, label);
//                maxTime = Math.max(maxTime, t);
//            }
//            nodeTimeline.put(i + 1, timeline); // 1-based node index
//        }
//
//        List<GraphSample> snapshots = new ArrayList<>();
//        Random random = new Random(42);
//
//        for (int t = 0; t <= maxTime; t++) {
//            // Group nodes by graphs at time t
//            Map<Integer, GraphSample> graphs = new HashMap<>();
//            Map<Integer, Integer> nodeGlobalToLocal = new HashMap<>();
//
//            int nodeLocalId = 0;
//            for (Map.Entry<Integer, TreeMap<Integer, Integer>> entry : nodeTimeline.entrySet()) {
//                int nodeId = entry.getKey();
//                TreeMap<Integer, Integer> timeline = entry.getValue();
//                Map.Entry<Integer, Integer> floor = timeline.floorEntry(t);
//                if (floor == null) continue; // Node doesn't exist at this time
//
//                int graphId = nodeToGraph.get(nodeId);
//                int graphLabel = graphLabels.get(graphId);
//
//                GraphSample graph = graphs.computeIfAbsent(graphId, gid -> new GraphSample(gid, graphLabel));
//                TsetlinVertex vertex = new TsetlinVertex("v" + nodeId, nodeLocalId++);
//                vertex.initializeLayerData(SparseBinaryVector.oneSegmentHot(floor.getValue()));
//                graph.vertices.add(vertex);
//                nodeGlobalToLocal.put(nodeId, nodeLocalId);
//            }
//
//            int edgeIdx = 0;
//            for (int[] edge : edgeList) {
//                int src = edge[0];
//                int dst = edge[1];
//                int availTime = edgeAvailableAt.getOrDefault(edgeIdx, Integer.MAX_VALUE);
//                if (availTime > t) {
//                    edgeIdx++;
//                    continue;
//                }
//
//                int g1 = nodeToGraph.get(src);
//                int g2 = nodeToGraph.get(dst);
//                if (g1 != g2 || !graphs.containsKey(g1)) {
//                    edgeIdx++;
//                    continue;
//                }
//
//                GraphSample graph = graphs.get(g1);
//                if (!nodeGlobalToLocal.containsKey(src) || !nodeGlobalToLocal.containsKey(dst)) {
//                    edgeIdx++;
//                    continue;
//                }
//
//                int localSrc = nodeGlobalToLocal.get(src);
//                int localDst = nodeGlobalToLocal.get(dst);
//                float importance = 0.01f + 0.01f * random.nextFloat();
//                SparseBinaryVector encoder = (src < dst) ? rightEdgeEncoder : leftEdgeEncoder;
//                TsetlinEdge edgeObj = new TsetlinEdge("e" + edgeIdx, importance, encoder);
//                graph.edges.add(new GraphSample.TsetlinEdgeRecord(localSrc, localDst, edgeObj));
//                edgeIdx++;
//            }
//
//            // Assign time to each graph and add to snapshot list
//            for (GraphSample g : graphs.values()) {
//                g.setTime(t);
//                snapshots.add(g);
//            }
//        }
//
//        return snapshots;
//    }
//
//




    private static Map<Integer, Integer> loadGraphIndicators(Path file) throws IOException {
        List<String> lines = Files.readAllLines(file);
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < lines.size(); i++) {
            map.put(i + 1, Integer.parseInt(lines.get(i).trim())); // 1-based node index
        }
        return map;
    }

    private static Map<Integer, Integer> loadGraphLabels(Path file) throws IOException {
        List<String> lines = Files.readAllLines(file);
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < lines.size(); i++) {
            map.put(i + 1, Integer.parseInt(lines.get(i).trim())); // 1-based graph index
        }
        return map;
    }

    private static Map<Integer, Integer> tryLoadNodeLabels(Path file) throws IOException {
        if (!Files.exists(file)) return null;
        List<String> lines = Files.readAllLines(file);
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < lines.size(); i++) {
            map.put(i + 1, Integer.parseInt(lines.get(i).trim())); // 1-based node index
        }
        return map;
    }

    private static Map<Integer, TreeMap<Integer, Integer>> tryLoadTemporalNodeLabels(Path file) throws IOException {
        if (!Files.exists(file)) return null;
        List<String> lines = Files.readAllLines(file);
        Map<Integer, TreeMap<Integer, Integer>> nodeTimeLabels = new HashMap<>();

        for (int i = 0; i < lines.size(); i++) {
            String[] tokens = lines.get(i).trim().split(",");
            TreeMap<Integer, Integer> timeSeries = new TreeMap<>();
            for (int j = 0; j < tokens.length - 1; j += 2) {
                int time = Integer.parseInt(tokens[j].trim());
                int label = Integer.parseInt(tokens[j + 1].trim());
                timeSeries.put(time, label);
            }
            nodeTimeLabels.put(i + 1, timeSeries); // 1-based index
        }
        return nodeTimeLabels;
    }

    private static List<int[]> loadEdges(Path file) throws IOException {
        List<String> lines = Files.readAllLines(file);
        List<int[]> edges = new ArrayList<>();
        for (String line : lines) {
            String[] tokens = line.trim().split(",");
            edges.add(new int[]{
                    Integer.parseInt(tokens[0].trim()),
                    Integer.parseInt(tokens[1].trim())
            });
        }
        return edges;
    }

    private static Map<Integer, double[]> tryLoadNodeAttributes(Path file) throws IOException {


        if (!Files.exists(file)) return null;

        int dimension = 0;
        List<String> lines = Files.readAllLines(file);
        Map<Integer, double[]> map = new HashMap<>();
        for (int i = 0; i < lines.size(); i++) {
            String[] tokens = lines.get(i).trim().split(",");
            double[] attr = new double[tokens.length];
            for (int j = 0; j < tokens.length; j++) {
                attr[j] = Double.parseDouble(tokens[j].trim());
            }
            map.put(i + 1, attr); // 1-based node ID
        }
        //set the dimension based on the first node's attributes
        if (!map.isEmpty()) {
            dimension = map.values().iterator().next().length;
        }

        //find the max and min values for each dimension
        double[] minValues = new double[dimension];
        double[] maxValues = new double[dimension];
        Arrays.fill(minValues, Double.MAX_VALUE);
        Arrays.fill(maxValues, Double.MIN_VALUE);
        for (double[] attrs : map.values()) {
            for (int j = 0; j < attrs.length; j++) {
                if (attrs[j] < minValues[j]) minValues[j] = attrs[j];
                if (attrs[j] > maxValues[j]) maxValues[j] = attrs[j];
            }
        }
        //normalize the data for each dimension between 0 and 1
        for (Map.Entry<Integer, double[]> entry : map.entrySet()) {
            double[] attrs = entry.getValue();
            for (int j = 0; j < attrs.length; j++) {
                if (maxValues[j] - minValues[j] != 0) {
                    attrs[j] = (attrs[j] - minValues[j]) / (maxValues[j] - minValues[j]);
                } else {
                    attrs[j] = 0.0; // If max and min are the same, set to 0
                }
            }
        }

        return map;
    }

    private static List<String> tryReadLines(Path file) throws IOException {
        return Files.exists(file) ? Files.readAllLines(file) : null;
    }
}