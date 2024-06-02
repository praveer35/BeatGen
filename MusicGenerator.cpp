#include <array>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <set>

/*class ChordGenerator {
private:
    std::map<std::string, std::vector<std::string>> chordProgressions;

public:
    ChordGenerator() {
        // Define some basic chord progressions for major and minor keys
        chordProgressions["C Major"] = {"C", "Dm", "Em", "F", "G", "Am", "Bdim"};
        chordProgressions["A Minor"] = {"Am", "Bdim", "C", "Dm", "Em", "F", "G"};
        // You can define more chord progressions for other keys if needed
    }

    std::vector<std::string> generateChordProgression(const std::string& key) {
        if (chordProgressions.find(key) != chordProgressions.end()) {
            return chordProgressions[key];
        } else {
            std::cerr << "Chord progression for key " << key << " not found." << std::endl;
            return {}; // Return an empty vector if key not found
        }
    }
};*/

#define KEYS_LEN (24)
#define NOTES_LEN (12)
#define SCALE_LEN (7)




/*std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);*/

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine gen(seed);
std::uniform_real_distribution<double> dis (0.0, 1.0);

void printVec(std::vector<int> vec) {
	//std::cout << name << ":";
	for (auto &i : vec) {
		std::cout << " " << i;
	}
	std::cout << std::endl;
}

class ChordGenerator {

private:
	std::array<std::string, KEYS_LEN> keys = {{
		"C min", "C Maj", "C# min", "C# Maj",
		"D min", "D Maj", "D# min", "D# Maj",
		"E min", "E Maj", "F min", "F Maj",
		"F# min", "F# Maj", "G min", "G Maj",
		"G# min", "G# Maj", "A min", "A Maj",
		"A# min", "A# Maj", "B min", "B Maj"
	}};
	std::array<std::string, NOTES_LEN> notes = {{
		"C", "C#", "D", "D#", "E", "F",
		"F#", "G", "G#", "A", "A#", "B"
	}};
	std::array<double, SCALE_LEN> source_outgoing = {
		0.1, 0.05, 0.15, 0, 0.2, 0.2, 0.3
	};
	/*
	std::array<std::array<double, SCALE_LEN>, SCALE_LEN> pFSM = {{
		{0, 0, 0.1, 0.06, 0.24, 0.32, 0.28},
		{0, 0, 0.22, 0.17, 0.22, 0.11, 0.28},
		{0, 0, 0, 0, 0.5, 0, 0.5},
		{0, 0.2, 0, 0, 0.3, 0, 0.5},
		{0.2, 0.2, 0, 0.15, 0, 0.25, 0.2},
		{0.41, 0.14, 0.19, 0.06, 0.1, 0, 0.1},
		{0.12, 0.15, 0.15, 0.1, 0.27, 0.21, 0}
	}};
	*/
	std::array<std::array<double, SCALE_LEN>, SCALE_LEN> pFSM = {{
		{0, 0, 0.24, 0.06, 0.12, 0.16, 0.42},
		{0, 0, 0.22, 0.17, 0.22, 0.11, 0.28},
		{0, 0, 0, 0, 0.5, 0, 0.5},
		{0, 0.2, 0, 0, 0.3, 0, 0.5},
		{0.1, 0.2, 0, 0.25, 0, 0.15, 0.3},
		{0.1, 0.14, 0.19, 0.17, 0.1, 0, 0.3},
		{0.12, 0.15, 0.15, 0.1, 0.27, 0.21, 0}
	}};
	int key;
	int bars;

	double rand_uniform(double low, double high) {
		return (dis(gen) * (high - low) + low);
	}

public:
	ChordGenerator(int key, int bars) {
		this->key = key;
		this->bars = bars;
	}
	
	int convertChord(int chord) {
		int new_chord = 0;
		switch (chord) {
			case 0: new_chord = key;                        break;
			case 1: new_chord = (key + 3) % keys.size();    break;
			case 2: new_chord = (key + 7) % keys.size();    break;
			case 3: new_chord = (key + 8) % keys.size();    break;
			case 4: new_chord = (key + 10) % keys.size();   break;
			case 5: new_chord = (key + 14) % keys.size();   break;
			case 6: new_chord = (key + 17) % keys.size();
		}
		return new_chord;
	}

	std::vector<int> generateChords() {
		int n = SCALE_LEN;
		int k = bars;

		double source_probability = rand_uniform(0, 1);
		int source;
		for (source = 0; source < SCALE_LEN; source++) {
			source_probability -= source_outgoing[source];
			if (source_probability < 0) {
				break;
			}
		}
		
		std::vector<int> row(k, -1);
		std::vector<std::vector<int>> edges(n, row);
		std::vector<double> vertex_probabilities(n, 0);
		std::set<int> iter_set;

		int iter = 0;
		for (int i = 0; i < n; i++) {
			if (pFSM[i][source]) {
				vertex_probabilities[i] = pFSM[i][source];
				edges[i][iter] = 0;
				iter_set.insert(i);
			}
		}

		iter++;
		std::vector<double> temp_vertex_probabilities(n, -1);
		
		while (iter < k - 1) {
			std::set<int> temp_set;

			for (auto &entry : iter_set) {
				for (int i = 0; i < n; i++) {
					if (pFSM[i][entry] && !temp_set.count(i)) {
						temp_set.insert(i);
						std::vector<double> chosen_vertex_probabilities(n, 0);
						double total_probability = 0;
						for (int j = 0; j < n; j++) {
							chosen_vertex_probabilities[j] = vertex_probabilities[j] * pFSM[i][j];
							total_probability += chosen_vertex_probabilities[j];
						}
						double rand = rand_uniform(0, total_probability);
						int j;
						for (j = 0; j < n; j++) {
							rand -= chosen_vertex_probabilities[j];
							if (rand < 0) {
								break;
							}
						}
						edges[i][iter] = j;
						temp_vertex_probabilities[i] = chosen_vertex_probabilities[j];
					}
				}
			}
			iter_set.clear();
			iter_set = temp_set;
			for (int i = 0; i < n; i++) {
				if (temp_vertex_probabilities[i] != -1) {
					vertex_probabilities[i] = temp_vertex_probabilities[i];
				}
			}

			iter++;
		}

		std::vector<double> source_neighbor_probabilities(n, 0);
		double total_probability = 0;
		for (auto &entry : iter_set) {
			source_neighbor_probabilities[entry] = pFSM[source][entry] * vertex_probabilities[entry];
			total_probability += source_neighbor_probabilities[entry];
		}
		double rand = rand_uniform(0, total_probability);
		int j;
		for (j = 0; j < n; j++) {
			rand -= source_neighbor_probabilities[j];
			if (rand < 0) {
				break;
			}
		}
		edges[source][iter] = j;

		iter++;

		std::vector<int> progression;

		// NOTE: REMOVE LATER //
		if (source < 3) {
			progression.push_back(source+1);
		} else {
			progression.push_back(source);
		}
		for (int i = 0; i < k - 1; i++) {
			source = edges[source][k - i - 1];


			// NOTE: REMOVE LATER //
			if (source < 3) {
				progression.push_back(source+1);
			} else {
				progression.push_back(source);
			}


			//progression.push_back(source);
		}

		return progression;

		/*std::vector<std::string> chord_list;

		for (auto &i : progression) {
			chord_list.push_back(keys[convertChord(i)]);
		}

		return chord_list;*/

		//printVec("progression", progression);
	}

	/*std::vector<std::string> generateChords() {
		std::vector<std::string> chord_list;
		int chord = 0;
		int first_chord = 0;
		double rand_num;
		int j;
		for (int i = 0; i < bars; i++) {
			rand_num = rand_uniform(0, 1);
			std::cout << rand_num << std::endl;
			for (j = 0; rand_num >= 0; j++) {
				rand_num -= pFSM[chord][j];
			}
			chord = j - 1;
			if (i == 0) {
				first_chord = chord;
			}
			if (i == bars - 1 && pFSM[chord][first_chord] == 0) {
				if (pFSM[chord][5] != 0) {
					chord = 5;
				} else if (pFSM[chord][6] != 0) {
					chord = 6;
				} else {
					chord = 4;
				}
			}
			chord_list.push_back(keys[convertChord(chord)]);
			//chord_list[i].assign keys[convertChord(chord)];
		}
		return chord_list;
	}*/
};

bool contains_duplicate(std::vector<int> vec) {
	std::set<int> myset;
	for (auto &i : vec) {
		if (myset.find(i) != myset.end()) { return true; }
		myset.insert(i);
	}
	return false;
}

int main() {
	/*std::vector<int> vec = {4, 3, 6, 3};
	if (contains_duplicate(vec)) { std::cout << "true\n"; }
	else { std::cout << "false\n"; }
	return 0;*/
	int key = 0;
	int bars = 0;
	//std::cin >> key >> bars;
	key = 1;
	bars = 4;
	ChordGenerator chordGen(key, bars);
	std::vector<int> chord_list;
	while (true) {
		chord_list = chordGen.generateChords();
		//printVec(chord_list);
		if (!contains_duplicate(chord_list)) { break; }
	}
	//std::vector<std::string> chord_list = chordGen.generateChords();
	for (const auto& chord : chord_list) {
		//if (chord < 3) std::cout << (chord+1) << " ";
		std::cout << chord << " ";
	}

	return 0;
}