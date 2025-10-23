#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include "Tokenizer.h"

namespace fs = std::filesystem;

std::vector<std::string> loadAllTexts(const std::string& folder) {
    std::vector<std::string> texts;
    for (const auto& entry : fs::recursive_directory_iterator(folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::ifstream in(entry.path());
            if (!in) continue;

            std::string content((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
            texts.push_back(content);
            std::cout << "Učitano: " << entry.path().string() << std::endl;
        }
    }
    std::cout << "Ukupno fajlova: " << texts.size() << std::endl;
    return texts;
}

int main() {
    std::cout << "=== DaraGPT Tokenizer Trainer ===" << std::endl;

    std::string dataDir = "./Data";
    if (!fs::exists(dataDir)) {
        std::cerr << "Direktorijum 'Data' ne postoji!" << std::endl;
        return 1;
    }

    // učitaj sve tekstove
    auto texts = loadAllTexts(dataDir);
    if (texts.empty()) {
        std::cerr << "Nema .txt fajlova u " << dataDir << std::endl;
        return 1;
    }

    // treniraj tokenizer
    Tokenizer tokenizer;
    tokenizer.TrainBPE(texts, 50000); // ili više tokena po želji

    // sačuvaj model
    tokenizer.Save("./checkpoints/tokenizer.tokbin");

    std::cout << "\nTrening završen i fajl sačuvan.\n";
    return 0;
}
