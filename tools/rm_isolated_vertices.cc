#include <iostream>
#include <fstream>
#include <sstream> 
#include <vector>
#include <string>
#include <cassert>

using nodeId = u_int32_t;
using edgeId = u_int64_t;

using namespace std;

string GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

int main(int argc, char** argv)
{
    if(argc!= 2)
	{
		cout << "\nThere was an error parsing command line arguments\n";
		exit(0);
	}

    string input = string(argv[1]);

    if(GetFileExtension(input) != "edges") {
        cout << "\nInput file format is not supported.\n";
		exit(0);
    }

    ifstream infile;
    infile.open(input);
    stringstream ss;
    u_int64_t max = 0;
    string line;
    edgeId edgeCounter = 0;
    
    vector<pair<u_int64_t, u_int64_t>> rawEdgeList;
    u_int64_t source;
    u_int64_t end;

    while(getline( infile, line ))
    {
        ss.str("");
        ss.clear();
        ss << line;

        if (!isdigit(ss.peek())) {
            continue;
        }
        
        ss >> source;
        ss >> end;

        rawEdgeList.emplace_back(source, end);
        
        if(max < source)
            max = source;
        if(max < end)
            max = end;				
    }
    infile.close();

    cout << "max node id: " << max << endl;

    bool *existed = new bool[max + 1];
    for (u_int64_t i = 0; i <= max; i++) {
        existed[i] = false;
    }

    for (auto& e: rawEdgeList) {
        existed[e.first] = true;
        existed[e.second] = true;
    }

    nodeId *nodeMap = new nodeId[max + 1];
    u_int64_t nonzero_deg_count = 0;
    for (u_int64_t i = 0; i <= max; i++) {
        nodeMap[i] = nonzero_deg_count;
        if (existed[i]) {
            nonzero_deg_count++;
        }
    }

    delete[] existed;

    cout << "num nodes: " << nonzero_deg_count << endl;

    std::ofstream outfile(input.substr(0, input.length()-5)+"el");
    for (auto& e: rawEdgeList) {
        outfile << nodeMap[e.first] << " " << nodeMap[e.second] << endl;
    }
    outfile.close();

    delete[] nodeMap;

    return 0;
}
