#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <string>
#include <algorithm>

using namespace std;

struct Record {
    map<string, string> features;
    string label;
};


double entropy(const vector<Record>& data) {
    map<string, int> labelCount;
    for (auto& record : data) {
        labelCount[record.label]++;
    }

    double entropyValue = 0.0;
    int total = data.size();
    for (auto& labelPair : labelCount) {
        double probability = (double)labelPair.second / total;
        entropyValue -= probability * log2(probability);
    }
    return entropyValue;
}


string bestSplit(const vector<Record>& data, const vector<string>& featureNames) {
    double baseEntropy = entropy(data);
    double bestGain = 0;
    string bestFeature = "";

    for (auto& feature : featureNames) {
        map<string, vector<Record>> subsets;
        for (auto& record : data) {
            subsets[record.features.at(feature)].push_back(record);
        }

        double newEntropy = 0;
        for (auto& subset : subsets) {
            double probability = (double)subset.second.size() / data.size();
            newEntropy += probability * entropy(subset.second);
        }

        double infoGain = baseEntropy - newEntropy;
        if (infoGain > bestGain) {
            bestGain = infoGain;
            bestFeature = feature;
        }
    }

    return bestFeature;
}


struct Node {
    string feature;
    string label;
    map<string, Node*> children;
};

Node* buildTree(const vector<Record>& data, vector<string> featureNames) {
    set<string> uniqueLabels;
    for (auto& record : data) {
        uniqueLabels.insert(record.label);
    }

    if (uniqueLabels.size() == 1) {
        return new Node{ "", *uniqueLabels.begin(), {} };
    }

    if (featureNames.empty()) {
        map<string, int> labelCount;
        for (auto& record : data) {
            labelCount[record.label]++;
        }
        string majorityLabel = max_element(labelCount.begin(), labelCount.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            })->first;
        return new Node{ "", majorityLabel, {} };
    }

    string bestFeature = bestSplit(data, featureNames);
    Node* rootNode = new Node{ bestFeature, "", {} };

    map<string, vector<Record>> subsets;
    for (auto& record : data) {
        subsets[record.features.at(bestFeature)].push_back(record);
    }

    vector<string> remainingFeatures;
    for (auto& feature : featureNames)
        if (feature != bestFeature) remainingFeatures.push_back(feature);

    for (auto& subset : subsets) {
        rootNode->children[subset.first] = buildTree(subset.second, remainingFeatures);
    }

    return rootNode;
}


void printTree(Node* rootNode, string indent = "") {
    if (!rootNode->feature.empty()) {
        for (auto& child : rootNode->children) {
            cout << indent << rootNode->feature << " = " << child.first << ":\n";
            printTree(child.second, indent + "  ");
        }
    } else {
        cout << indent << "=> " << rootNode->label << "\n";
    }
}



void generateDOT(Node* rootNode, ofstream& outFile, map<Node*, int>& nodeIds, int& nextId) {
    if (rootNode == nullptr) return;

    int currentId = nextId++;
    nodeIds[rootNode] = currentId;

    if (rootNode->feature.empty()) {
        // Leaf node
        outFile << "    node" << currentId << " [label=\"Label: " << rootNode->label << "\", shape=box];\n";
    } else {
        // Decision node
        outFile << "    node" << currentId << " [label=\"" << rootNode->feature << "\"];\n";
        for (auto& child : rootNode->children) {
            int childId = nextId;
            generateDOT(child.second, outFile, nodeIds, nextId);
            childId = nodeIds[child.second];
            outFile << "    node" << currentId << " -> node" << childId << " [label=\"" << child.first << "\"];\n";
        }
    }
}

vector<Record> readCSV(const string& filename, vector<string>& featureNames) {
    vector<Record> data;
    ifstream file(filename);
    string line;

    if (!getline(file, line)) {
        cerr << "Error reading header.\n";
        return data;
    }

    stringstream ss(line);
    string item;
    vector<string> headers;

    while (getline(ss, item, ',')) {
        headers.push_back(item);
    }

    featureNames.assign(headers.begin(), headers.end() - 1);
    string labelCol = headers.back();

    while (getline(file, line)) {
        stringstream row(line);
        Record rec;
        for (size_t i = 0; i < headers.size(); i++) {
            getline(row, item, ',');
            if (i == headers.size() - 1)
                rec.label = item;
            else
                rec.features[headers[i]] = item;
        }
        data.push_back(rec);
    }

    return data;
}

int main() {
    vector<string> featureNames;
    vector<Record> dataset = readCSV("TBD3.csv", featureNames);

    if (dataset.empty()) {
        cerr << "Dataset is empty or failed to load.\n";
        return 1;
    }

    Node* tree = buildTree(dataset, featureNames);

    ofstream outFile("tree.dot");
    outFile << "digraph G {\n";

    map<Node*, int> nodeIds;
    int nextId = 0;
    generateDOT(tree, outFile, nodeIds, nextId);

    outFile << "}\n";
    outFile.close();

    cout << "DOT file created successfully as 'tree.dot'.\n";

    return 0;
}
