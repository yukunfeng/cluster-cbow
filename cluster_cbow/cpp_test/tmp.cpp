#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <map>

using namespace std;

void readCluster(std::istream& in) {
    std::string line;
    std::map<string,string> word2cluster;
    while(std::getline(in, line)) {
        // std::cout << line << std::endl;
        std::stringstream linestream(line);
        std::string val1;
        std::string val2;
        // std::getline(linestream, data, ' ');

        linestream >> val1 >> val2;
        cout << val1 << ":" << val2 << endl;
        word2cluster[val1] = val2;
    }
    cout << word2cluster["1"] << "hahah" << endl;
    if (!word2cluster.empty()) {
        cout << "map size: "  << word2cluster.size() << endl;
    }
}

int main(int argc, char *argv[])
{
    // std::string path = "./tmp.txt";
    // std::ifstream ifs(path.c_str());
    // readCluster(ifs);
    // ifs.close();
    //
    string t = "<ok12>";
    cout << t.substr(1, t.size() - 2) << endl;
    return 0;
}
