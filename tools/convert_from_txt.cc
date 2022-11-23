#include <iostream>
#include <fstream>
#include <sstream> 
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>

using nodeId = u_int32_t;
using edgeId = u_int64_t;

using namespace std;

struct OutEdge{
    nodeId end;
};

struct Edge{
	nodeId source;
    nodeId end;
};

string GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

bool test_neighbor(nodeId u, nodeId v, edgeId *rowptr, nodeId *col) {
	bool neighbor = false;

	for (edgeId e = rowptr[u]; e < rowptr[u + 1]; e++) {
		if (col[e] == v) {
			neighbor = true;
		}
	}

	if (!neighbor) {
		cout << "edge (" << u << " , " << v << " ) not existed!" << endl;
		return false;
	}
	return true;
}

bool test(vector<nodeId> &sources, vector<nodeId> &ends, edgeId *rowptr, nodeId *col) {
	if (sources.size() != ends.size()) {
		cout << "wrong edgelist size" << endl;
		return false;
	}

	for (edgeId i = 0; i < sources.size(); i++) {
		bool valid = true;
		valid &= test_neighbor(sources[i], ends[i], rowptr, col);
		valid &= test_neighbor(ends[i], sources[i], rowptr, col);

		if (!valid) {
			exit(1);
		}
	}

	cout << "ok" << endl;
	return true;
}

void read_txt(string &input, vector<pair<u_int64_t, u_int64_t>> &rawEdgeList, size_t &max) {
	ifstream infile;
	infile.open(input);
	stringstream ss;
	string line;

	u_int64_t source;
	u_int64_t end;

	cout << "read graph in text file..." << endl;

	while(getline( infile, line ))
	{
		ss.str("");
		ss.clear();
		ss << line;

		if (ss.peek() < '0' || ss.peek() > '9') {
			continue;
		}

		ss >> source;
		ss >> end;

		// remove self loop
		if (source == end) {
			continue;
		}

		rawEdgeList.emplace_back(source, end);

		if(max < source)
			max = source;
		if(max < end)
			max = end;
	}
	infile.close();
}

void remove_isolated_vertices(vector<pair<u_int64_t, u_int64_t>> &rawEdgeList, const size_t max) {
	cout << "remove isolated vertices, max node id: " << max << " ..." << endl;

	bool *existed = new bool[max + 1];
	for (size_t i = 0; i <= max; i++) {
		existed[i] = false;
	}

    for (auto& e: rawEdgeList) {
        existed[e.first] = true;
        existed[e.second] = true;
    }

	cout << "[remove isolated vertices] set existed vertices done" << endl;

    size_t *nodeMap = new size_t[max + 1];
    size_t nonzero_deg_count = 0;
    for (size_t i = 0; i <= max; i++) {
        nodeMap[i] = nonzero_deg_count;
        if (existed[i]) {
            nonzero_deg_count++;
        }
    }
	delete[] existed;

	cout << "[remove isolated vertices] create map" << endl;

	for (auto& e: rawEdgeList) {
		e.first = nodeMap[e.first];
		e.second = nodeMap[e.second];
    }

	delete[] nodeMap;
}

void to_undirected(vector<pair<u_int64_t, u_int64_t>> &rawEdgeList, vector<pair<nodeId, nodeId>> &edgeList) {
	cout << "convert into undirected graph..." << endl;

	edgeId numEdge = rawEdgeList.size();
	for (edgeId i = 0; i < numEdge; i++) {
		rawEdgeList.emplace_back(rawEdgeList[i].second, rawEdgeList[i].first);
	}

	sort(rawEdgeList.begin(), rawEdgeList.end(), [](const pair<u_int64_t, u_int64_t> &left, const pair<u_int64_t, u_int64_t>& right) {
		if (left.first == right.first) {
			return left.second < right.second;
		}
		return left.first < right.first;
	});

	pair<u_int64_t, u_int64_t> prev_edge = make_pair(0, 0);
	for (const auto &edge: rawEdgeList) {
		if (prev_edge != edge) {
			prev_edge = edge;
			edgeList.emplace_back(edge.first, edge.second);
		}
	}
}

int main(int argc, char** argv)
{
	if(argc!= 2)
	{
		cout << "\nThere was an error parsing command line arguments\n";
		exit(0);
	}
	
	string input = string(argv[1]);
	
	if(GetFileExtension(input) == "el")
	{

		vector<pair<nodeId, nodeId>> edgeList;

		{
			vector<pair<u_int64_t, u_int64_t>> rawEdgeList;
			size_t max = 0;

			read_txt(input, rawEdgeList, max);
			remove_isolated_vertices(rawEdgeList, max);
			to_undirected(rawEdgeList, edgeList);
		}

		nodeId num_nodes = edgeList.back().first + 1;
		edgeId num_edges = edgeList.size();

		cout << "nodes: " << num_nodes << " edges: " << num_edges << endl;


		vector<edgeId> nodePointer(num_nodes + 1, 0);
		for(edgeId i = 0; i < num_edges; i++)
		{
			nodePointer[edgeList[i].first + 1]++;
		}		
		for(nodeId i = 0; i < num_nodes; i++)
		{
			nodePointer[i+1] += nodePointer[i];
		}

		vector<nodeId> edges;
		edges.reserve(num_edges);
		for(edgeId i=0; i<num_edges; i++)
		{
			edges.push_back(edgeList[i].second);
		}
		
		std::ofstream outfile(input.substr(0, input.length()-2)+"bcsr", std::ofstream::binary);
		
		outfile.write((char*)&num_nodes, sizeof(nodeId));
		outfile.write((char*)&num_edges, sizeof(edgeId));
		outfile.write ((char*)nodePointer.data(), sizeof(edgeId)*(num_nodes + 1));
		outfile.write ((char*)edges.data(), sizeof(OutEdge)*num_edges);
		
		outfile.close();
	}
	else
	{
		cout << "\nInput file format is not supported.\n";
		exit(0);
	}

}
