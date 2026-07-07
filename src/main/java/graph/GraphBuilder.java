package graph;

public class GraphBuilder {
    // Set of visited CIKs to prevent infinite recursion
//    private final Set<String> visited = ConcurrentHashMap.newKeySet();
//    private final EdgarClient edgar = new EdgarClient();
//
//    // Virtual Thread Executor: Spawns a new light thread for every task
//    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
//
//    //create a logger
//    private static final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(GraphBuilder.class);
//
//    public void buildGraph(String rootCik, int maxNodes) {
//        recursiveScrape(rootCik, maxNodes);
//    }
//
//    private void recursiveScrape(String cik, int limit) {
//        if (visited.size() >= limit || visited.contains(cik)) return;
//        visited.add(cik);
//
//        executor.submit(() -> {
//            try {
//                logger.info("Processing CIK: {}", cik);
//                String json = edgar.getCompanySubmissions(cik);
//
//                // 1. Convert to Hypervector (Your logic here)
//                // MyGraphModel.addNode(cik, convertToHypervector(json));
//
//                // 2. Find Subsidiaries (The Edges)
//                List<String> children = findSubsidiariesInFiling(json);
//
//                for (String childCik : children) {
//                    // Recursive call: Spawns another virtual thread
//                    recursiveScrape(childCik, limit);
//
//                    // MyGraphModel.addEdge(cik, childCik);
//                }
//            } catch (Exception e) {
//                logger.error("Failed to process {}: {}", cik, e.getMessage());
//            }
//        });
//    }


}