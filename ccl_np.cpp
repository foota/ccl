// Marathon Match - CCL - Neighbour Propagation

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <list>
#include <algorithm>
#include <utility>
#include <cmath>
#include <functional>
#include <cstring>
#include <cmath>
#include <limits>

#define NOMINMAX

#ifdef _MSC_VER
#include <ctime>
inline double get_time()
{
        return static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
}
#else
#include <sys/time.h>
inline double get_time()
{
        timeval tv;
        gettimeofday(&tv, 0);
        return tv.tv_sec + 1e-6 * tv.tv_usec;
}
#endif

using namespace std;

void init_CCL(int L[], int N)
{
	for (int id = 0; id < N; id++) L[id] = id;
}

inline int diff(int d1, int d2)
{
	return abs(((d1>>16) & 0xff) - ((d2>>16) & 0xff)) + abs(((d1>>8) & 0xff) - ((d2>>8) & 0xff)) + abs((d1 & 0xff) - (d2 & 0xff));
}

bool kernel(int D[], int L[], int N, int W, int th)
{
	bool m = false;

	for (int id = 0; id < N; id++) {
		int Did = D[id];
		int label = N;
		if (id - W >= 0 && diff(Did, D[id-W]) <= th) label = min(label, L[id-W]);
		if (id + W < N  && diff(Did, D[id+W]) <= th) label = min(label, L[id+W]);
		int r = id % W;
		if (r           && diff(Did, D[id-1]) <= th) label = min(label, L[id-1]);
		if (r + 1 != W  && diff(Did, D[id+1]) <= th) label = min(label, L[id+1]);

		if (label < L[id]) {
			L[id] = label;
			m = true;
		}
	}

	return m;
}

bool kernel8(int D[], int L[], int N, int W, int th)
{
	bool m = false;

	for (int id = 0; id < N; id++) {
		int Did = D[id];
		int label = N;
		if (id - W >= 0 && diff(Did, D[id-W]) <= th) label = min(label, L[id-W]);
		if (id + W < N  && diff(Did, D[id+W]) <= th) label = min(label, L[id+W]);
		int r = id % W;
		if (r) {
			if (diff(Did, D[id-1]) <= th) label = min(label, L[id-1]);
			if (id - W - 1 >= 0 && diff(Did, D[id-W-1]) <= th) label = min(label, L[id-W-1]);
			if (id + W - 1 < N  && diff(Did, D[id+W-1]) <= th) label = min(label, L[id+W-1]);
		}
		if (r + 1 != W) {
			if (diff(Did, D[id+1]) <= th) label = min(label, L[id+1]);
			if (id - W + 1 >= 0 && diff(Did, D[id-W+1]) <= th) label = min(label, L[id-W+1]);
			if (id + W + 1 < N  && diff(Did, D[id+W+1]) <= th) label = min(label, L[id+W+1]);
		}

		if (label < L[id]) {
			L[id] = label;
			m = true;
		}
	}

	return m;
}

class CCL {
public:
	vector<int> ccl(vector<int>& image, int W, int degree_of_connectivity, int threshold);
};

vector<int> CCL::ccl(vector<int>& image, int W, int degree_of_connectivity, int threshold)
{
	int* D = static_cast<int*>(&image[0]);
	int N = image.size();
	int* L = new int[N];

	init_CCL(L, N);

	for (bool m = true; m; m = degree_of_connectivity == 4 ? kernel(D, L, N, W, threshold) : kernel8(D, L, N, W, threshold));

	vector<int> result(L, L + N);

	delete [] L;

	return result;
}

void read_data(const string filename, vector<int>& image, int& W, int& degree_of_connectivity, int& threshold)
{
	fstream fs(filename.c_str(), ios_base::in);
	string line;
	stringstream ss;
	int data;

	getline(fs, line);
	ss.str(line);
	ss >> W >> degree_of_connectivity >> threshold;
	getline(fs, line);
	ss.str("");  ss.clear();
	for (ss.str(line); ss >> data; image.push_back(data));
}

int main(int argc, char* argv[])
{
	ios_base::sync_with_stdio(false);

	if (argc < 2) {
		cerr << "Usage: " << argv[0] << " input_file" << endl;
		exit(1);
	}

	vector<int> image;
	int W, degree_of_connectivity, threshold;
	read_data(argv[1], image, W, degree_of_connectivity, threshold);

	CCL ccl;

	double start = get_time();
	vector<int> result(ccl.ccl(image, W, degree_of_connectivity, threshold));
	double end = get_time();
	cerr << "Time: " << end - start << endl;

	cout << result.size() << endl; /// number of pixels
	cout << W << endl; /// width
	for (int i = 0; i < static_cast<int>(result.size()) / W; i++) {
		for (int j = 0; j < W; j++) cout << result[i*W+j] << " ";
		cout << endl;
	}

	return 0;
}
