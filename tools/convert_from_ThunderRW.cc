#include <iostream>
#include <fstream>
#include <sstream> 
#include <vector>
#include <string>
#include <cassert>

using nodeId = u_int32_t;
using edgeId = u_int64_t;

using namespace std;

void test(nodeId *edgelist, edgeId *rowptr, nodeId *col, nodeId numNode, edgeId numEdge) {
    for (edgeId e = 0; e < numEdge; e++) {
        nodeId u = edgelist[2 * e];
        nodeId v = edgelist[2 * e + 1];

        bool neighbor = false;
        for (nodeId i = rowptr[u]; i < rowptr[u + 1]; i++) {
            if (col[i] == v) {
                neighbor = true;
            }
        }

        assert(neighbor);
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
    string degFileName = input + "b_degree.bin";
    string adjFileName = input + "b_adj.bin";
    
    ifstream degFile;
    ifstream adjFile;
    degFile.open(degFileName);
    adjFile.open(adjFileName);

    nodeId numNode;
    edgeId numEdge;
    degFile.read((char *)&numNode, sizeof(numNode));
    degFile.read((char *)&numNode, sizeof(numNode));
    degFile.read((char *)&numEdge, sizeof(numEdge));
    

    cout << "num Node: " << numNode << ", num Edge: " << numEdge << endl;

    nodeId *rawDegree = new nodeId[numNode];
    nodeId *degree = new nodeId[numNode];
    degFile.read((char *)rawDegree, sizeof(rawDegree[0]) * numNode);
    degFile.close();

    nodeId nonzero_deg_count = 0;
    nodeId *nodeMap = new nodeId[numNode];

    for (nodeId i = 0; i < numNode; i++) {
        nodeMap[i] = nonzero_deg_count;
        if (rawDegree[i] > 0) {
            degree[nonzero_deg_count] = rawDegree[i];
            nonzero_deg_count++;
        }
    }

    cout << "node with 0 degree: " << numNode - nonzero_deg_count << endl;
    delete[] rawDegree;
    numNode = nonzero_deg_count;
    cout << "num Node: " << numNode << ", num Edge: " << numEdge << endl;

    edgeId *rowptr = new edgeId[numNode + 1];
    rowptr[0] = 0;
    for (nodeId i = 0; i < numNode; i++) {
        rowptr[i + 1] = rowptr[i] + degree[i];
    }

    delete[] degree;

    nodeId *col = new nodeId[numEdge];
    adjFile.read((char *)col, sizeof(nodeId) * numEdge);
    adjFile.close();

    for (edgeId i = 0; i < numEdge; i++) {
        col[i] = nodeMap[col[i]];
    }

    delete[] nodeMap;

    std::ofstream csrFile(input + "graph.bcsr", std::ofstream::binary);
    csrFile.write((char *)&numNode, sizeof(numNode));
    csrFile.write((char *)&numEdge, sizeof(numEdge));
    csrFile.write((char *)rowptr, sizeof(edgeId) * (numNode + 1));
    csrFile.write((char *)col, sizeof(nodeId) * numEdge);
    csrFile.close();

    cout << "convert csr file done!" << endl;

    nodeId *edgeList = new nodeId[numEdge * 2];
    nodeId n = 0;
    for (edgeId i = 0; i < numEdge; i++) {
        while (i >= rowptr[n + 1]) {
            n++;
        }

        edgeList[i * 2] = n;
        edgeList[i * 2 + 1] = col[i];
    }

    std::ofstream elFile(input + "graph.bel", std::ofstream::binary);
    elFile.write((char *)edgeList, sizeof(nodeId) * 2 * numEdge);
    elFile.close();

    cout << "convert edgelist file done!" << endl;
    
    delete[] rowptr;
    delete[] col;
    delete[] edgeList;
    return 0;
}
