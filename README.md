The whole process of deep matching for two  large graphes is divided into five stages as following:
	1. Detection the community in the  graphes
	2. Matching the communities by using their features 
	3. Pre-process stage : apply the nodes matching method, deepwalk&CPD or bipartitie_matching, to the subgraphes generated wiht the matched communities and obtain the  matched nodes pairs between the pairs of communities
	4. Edges-credibility stage : With the matched nodes pairs obtained in stage 3 as the input to this stage, using the edges credibility algritom to detect the real seed nodes pairs as the output.  
	5. Refine stage : With the real seed nodes pairs obtained in stage 4, applying the propagation algritom to match the rest nodes of the different subgraphes.
