#include "Tokenizer.h"
#include <unordered_map>
#include <string_view>
#include <numeric>

struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
        // jednostavan kombinujući hash
        std::hash<std::string> h;
        return h(p.first) ^ (h(p.second) + 0x9e3779b97f4a7c15ULL + (h(p.first) << 6) + (h(p.first) >> 2));
    }
};


Tokenizer::Tokenizer() {
    AddToken("<PAD>");
    AddToken("<UNK>");
    AddToken("<BOS>");
    AddToken("<EOS>");
}

int Tokenizer::AddToken(const std::string& token) {
    if (TokenToId.count(token)) return TokenToId[token];
    int id = nextId++;
    TokenToId[token] = id;
    IdToToken[id] = token;
    return id;
}

void Tokenizer::TrainBPE(const std::vector<std::string>& texts, int vocabTarget) {
    std::cout << "Treniram BPE tokenizator..." << std::endl;

    // 1) Keširamo sve "reči" kao vektore simbola, plus njihove frekvencije
    // ---------------------------------------------------------------------
    std::vector<std::vector<std::string>> vocabSymbols; // svaka "reč" je niz simbola
    std::vector<int>                        freqs;       // učestalost svake reči

    vocabSymbols.reserve(texts.size() * 4);
    freqs.reserve(texts.size() * 4);

    // privremeni indeks: "string sa razmacima" -> index u vocabSymbols
    // (samo prilikom prvog prolaza nad tekstom; posle radimo in-place)
    std::unordered_map<std::string, int> indexBySpaced;
    indexBySpaced.reserve(texts.size() * 8);

    auto to_spaced = [](const std::vector<std::string>& syms) -> std::string {
        std::string out;
        size_t total = 0;
        for (auto& s : syms) total += s.size() + 1;
        out.reserve(total);
        for (size_t i = 0; i < syms.size(); ++i) {
            if (i) out.push_back(' ');
            out.append(syms[i]);
        }
        return out;
    };

    for (const auto& text : texts) {
        std::istringstream ss(text);
        std::string word;
        while (ss >> word) {
            std::vector<std::string> chars;
            chars.reserve(word.size() + 1);
            for (char c : word) chars.emplace_back(1, c);
            chars.emplace_back("</w>");

            // sabiramo učestalosti na nivou "reči"
            std::string spaced = to_spaced(chars);
            auto it = indexBySpaced.find(spaced);
            if (it == indexBySpaced.end()) {
                int idx = (int)vocabSymbols.size();
                indexBySpaced.emplace(std::move(spaced), idx);
                vocabSymbols.push_back(std::move(chars));
                freqs.push_back(1);
            } else {
                freqs[it->second] += 1;
            }
        }
    }

    // mala zaštita: ako je korpus prazan
    if (vocabSymbols.empty()) {
        std::cout << "Nema podataka za trening." << std::endl;
        return;
    }

    // 2) Pomoćna lambda: prebroji sve parove (simbol_i, simbol_{i+1})
    //    Koristi se brza tabela (unordered_map) + thread-local akumulatori (po potrebi).
    // ---------------------------------------------------------------------
    auto count_all_pairs = [&](auto& outPairs) {
        outPairs.clear();
        // Heuristička rezervacija (smanjuje rehash)
        size_t approxPairs = 0;
        for (size_t i = 0; i < vocabSymbols.size(); ++i) {
            if (vocabSymbols[i].size() >= 2) approxPairs += (vocabSymbols[i].size() - 1);
        }
        outPairs.reserve(approxPairs * 2);

        // Sekvencijalno (brzo i bez OpenMP). Ako koristiš OpenMP, vidi napomenu ispod.
        for (size_t wi = 0; wi < vocabSymbols.size(); ++wi) {
            const auto& syms = vocabSymbols[wi];
            const int f = freqs[wi];
            for (size_t j = 0; j + 1 < syms.size(); ++j) {
                outPairs[{syms[j], syms[j + 1]}] += f;
            }
        }
    };

    // 3) Pomoćna funkcija: primeni merge (a,b) in-place nad svim "rečima"
    //    bez rekonstrukcije stringa; spajamo susedne simbole tamo gde se poklapaju.
    // ---------------------------------------------------------------------
    auto apply_merge_inplace = [&](const std::pair<std::string, std::string>& bestPair) {
        const std::string& A = bestPair.first;
        const std::string& B = bestPair.second;
        const std::string AB = A + B;

        for (auto& syms : vocabSymbols) {
            if (syms.size() < 2) continue;
            // jednim prolazom spajaj
            std::vector<std::string> merged;
            merged.reserve(syms.size());
            for (size_t i = 0; i < syms.size();) {
                if (i + 1 < syms.size() && syms[i] == A && syms[i + 1] == B) {
                    merged.push_back(AB);
                    i += 2;
                } else {
                    merged.push_back(std::move(syms[i]));
                    i += 1;
                }
            }
            syms = std::move(merged);
        }
    };

    // 4) Glavna BPE petlja: dok ne dostignemo target vokabular
    //    (Korišćenje lokalnog unordered_map za brzinu; pairFreq (std::map) punimo za log/kompat.)
    // ---------------------------------------------------------------------
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> fastPairs;
    fastPairs.reserve(1 << 18);

    while ((int)TokenToId.size() < vocabTarget) {
        // prebroj sve parove
        count_all_pairs(fastPairs);
        if (fastPairs.empty()) break;

        // nadji najbolji par
        auto bestIt = std::max_element(
            fastPairs.begin(), fastPairs.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        );
        const auto bestPair = bestIt->first;

        // zapamti merge redosled (isti format)
        merges.push_back(bestPair);

        // AddToken za novo spajanje (isti format)
        AddToken(bestPair.first + bestPair.second);

        // primeni merge in-place
        apply_merge_inplace(bestPair);

        // (opcionalno) ispis napretka
        if (TokenToId.size() % 500 == 0)
            std::cout << "  • BPE merge " << TokenToId.size() << " tokena" << std::endl;

        // ako želiš da zadržiš i "pairFreq" strukturu radi kompatibilnosti/loga:
        pairFreq.clear(); // std::map< pair<string,string>, int >
        // prelij iz fastPairs (ovo je samo O(N) kopija, minoran trošak naspram brojanja)
        for (const auto& kv : fastPairs) {
            pairFreq.emplace(kv.first, kv.second);
        }
        // (sledeći krug svakako će fastPairs biti ponovo popunjen iz vocabSymbols)
    }

    std::cout << "BPE trening završen. Ukupno " << TokenToId.size() << " tokena." << std::endl;
}


std::vector<int> Tokenizer::Encode(const std::string& text) {
    std::vector<int> ids;
    std::istringstream ss(text);
    std::string word;

    while (ss >> word) {
        std::vector<std::string> chars;
        for (char c : word) chars.push_back(std::string(1, c));
        chars.push_back("</w>");

        for (const auto& [a, b] : merges) {
            for (size_t i = 0; i + 1 < chars.size();) {
                if (chars[i] == a && chars[i + 1] == b) {
                    chars[i] = a + b;
                    chars.erase(chars.begin() + i + 1);
                } else ++i;
            }
        }

        for (auto& token : chars) {
            if (TokenToId.count(token))
                ids.push_back(TokenToId[token]);
            else
                ids.push_back(TokenToId["<UNK>"]);
        }
    }

    return ids;
}

std::string Tokenizer::Decode(const std::vector<int>& ids) {
    std::ostringstream oss;
    for (int id : ids) {
        if (IdToToken.count(id))
            oss << IdToToken[id].substr(0, IdToToken[id].find("</w>")) << " ";
        else
            oss << "<UNK> ";
    }
    return oss.str();
}

void Tokenizer::Save(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return;

    out << "TOK" << std::endl;
    out << TokenToId.size() << std::endl;
    for (auto& kv : TokenToId)
        out << kv.first << " " << kv.second << std::endl;

    out << IdToToken.size() << std::endl;
    for (auto& kv : IdToToken)
        out << kv.first << " " << kv.second << std::endl;

    out << merges.size() << std::endl;
    for (auto& p : merges)
        out << p.first << " " << p.second << std::endl;

    out << nextId << std::endl;
    std::cout << "Tokenizer sačuvan BIN (" << TokenToId.size() << " tokena, "
              << merges.size() << " spajanja) → " << path << std::endl;
}

void Tokenizer::Load(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cout << "Tokenizer fajl nije pronađen: " << path << std::endl;
        return;
    }

    std::string magic;
    in >> magic;
    if (magic != "TOK") throw std::runtime_error("Nije TOK fajl.");

    size_t t2iCount;
    in >> t2iCount;
    TokenToId.clear();
    for (size_t i = 0; i < t2iCount; ++i) {
        std::string key;
        int val;
        in >> key >> val;
        TokenToId[key] = val;
    }

    size_t i2tCount;
    in >> i2tCount;
    IdToToken.clear();
    for (size_t i = 0; i < i2tCount; ++i) {
        int key;
        std::string val;
        in >> key >> val;
        IdToToken[key] = val;
    }

    size_t mCount;
    in >> mCount;
    merges.clear();
    for (size_t i = 0; i < mCount; ++i) {
        std::string a, b;
        in >> a >> b;
        merges.emplace_back(a, b);
    }

    in >> nextId;
    std::cout << "Tokenizer učitan BIN (" << TokenToId.size() << " tokena, "
              << merges.size() << " spajanja) ← " << path << std::endl;
}
