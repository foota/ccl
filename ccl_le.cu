// Marathon Match - CCL - Label Equivalence

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

#include <cutil_inline.h>

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

//const int BLOCK = 128;
const int BLOCK = 256;

__global__ void init_CCL(int L[], int R[], int N)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	L[id] = R[id] = id;
}

__device__ int diff(int d1, int d2)
{
	return abs(((d1>>16) & 0xff) - ((d2>>16) & 0xff)) + abs(((d1>>8) & 0xff) - ((d2>>8) & 0xff)) + abs((d1 & 0xff) - (d2 & 0xff));
}

__global__ void scanning(int D[], int L[], int R[], bool* m, int N, int W, int th)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	int Did = D[id];
	int label = N;
	if (id - W >= 0 && diff(Did, D[id-W]) <= th) label = min(label, L[id-W]);
	if (id + W < N  && diff(Did, D[id+W]) <= th) label = min(label, L[id+W]);
	int r = id % W;
	if (r           && diff(Did, D[id-1]) <= th) label = min(label, L[id-1]);
	if (r + 1 != W  && diff(Did, D[id+1]) <= th) label = min(label, L[id+1]);

	if (label < L[id]) {
		//atomicMin(&R[L[id]], label);
		R[L[id]] = label;
		*m = true;
	}
}

__global__ void scanning8(int D[], int L[], int R[], bool* m, int N, int W, int th)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

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
		//atomicMin(&R[L[id]], label);
		R[L[id]] = label;
		*m = true;
	}
}

__global__ void analysis(int D[], int L[], int R[], int N)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	int label = L[id];
	int ref;
	if (label == id) {
		do { label = R[ref = label]; } while (ref ^ label);
		R[id] = label;
	}
}

__global__ void labeling(int D[], int L[], int R[], int N)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	L[id] = R[R[L[id]]];
}

class CCL {
private:
	int* Dd;
	int* Ld;
	int* Rd;

public:
	vector<int> cuda_ccl(vector<int>& image, int W, int degree_of_connectivity, int threshold);
};

vector<int> CCL::cuda_ccl(vector<int>& image, int W, int degree_of_connectivity, int threshold)
{
	vector<int> result;
	int* D = static_cast<int*>(&image[0]);
	int N = image.size();

	cudaMalloc((void**)&Ld, sizeof(int) * N);
	cudaMalloc((void**)&Rd, sizeof(int) * N);
	cudaMalloc((void**)&Dd, sizeof(int) * N);
	cudaMemcpy(Dd, D, sizeof(int) * N, cudaMemcpyHostToDevice);

	bool* md;
	cudaMalloc((void**)&md, sizeof(bool));

	int width = static_cast<int>(sqrt(static_cast<double>(N) / BLOCK)) + 1;
	dim3 grid(width, width, 1);
	dim3 threads(BLOCK, 1, 1);

	init_CCL<<<grid, threads>>>(Ld, Rd, N);

	for (;;) {
		bool m = false;
		cudaMemcpy(md, &m, sizeof(bool), cudaMemcpyHostToDevice);
		if (degree_of_connectivity == 4) scanning<<<grid, threads>>>(Dd, Ld, Rd, md, N, W, threshold);
		else scanning8<<<grid, threads>>>(Dd, Ld, Rd, md, N, W, threshold);
		cudaMemcpy(&m, md, sizeof(bool), cudaMemcpyDeviceToHost);
		if (m) {
			analysis<<<grid, threads>>>(Dd, Ld, Rd, N);
			//cudaThreadSynchronize();
			labeling<<<grid, threads>>>(Dd, Ld, Rd, N);
		} else break;
	}

	cudaMemcpy(D, Ld, sizeof(int) * N, cudaMemcpyDeviceToHost);

	cudaFree(Dd);
	cudaFree(Ld);
	cudaFree(Rd);

	result.swap(image);
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

	cudaSetDevice(cutGetMaxGflopsDeviceId());

	vector<int> image;
	int W, degree_of_connectivity, threshold;
	read_data(argv[1], image, W, degree_of_connectivity, threshold);

	CCL ccl;

	double start = get_time();
	vector<int> result(ccl.cuda_ccl(image, W, degree_of_connectivity, threshold));
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
