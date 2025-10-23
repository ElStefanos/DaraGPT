#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

class Tokenizer {
private:
    std::unordered_map<std::string, int> TokenToId;
    std::unordered_map<int, std::string> IdToToken;
    std::vector<std::pair<std::string, std::string>> merges;
    std::map<std::pair<std::string, std::string>, int> pairFreq;
    int nextId = 0;

public:
    Tokenizer();

    int AddToken(const std::string& token);
    void TrainBPE(const std::vector<std::string>& texts, int vocabTarget = 150000);
    std::vector<int> Encode(const std::string& text);
    std::string Decode(const std::vector<int>& ids);

    void Save(const std::string& path);
    void Load(const std::string& path);
};
