#include <iostream>
#include <utility>
#include <chrono>
#include <cstring>

#include "../include/kdtree_index.hpp"

using namespace kdtree_index;

struct pod { int a; int b; };
struct ac_pod
{
	bool operator()(dimension_type d, const pod& a, const pod& b) const noexcept
	{ return (d == 0) ? a.a < b.a : a.b < b.b; }
};
typedef indexable<pod, 2, ac_pod> my_indexable;


int main (int, char **, char **)
{
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> elapsed_seconds;

	constexpr int Max = 100000;
	kdtree<my_indexable> my_tree(Max);

	start = std::chrono::system_clock::now();

	for (int i = 0; i < Max; ++i) my_tree.insert({i, Max - i});

	end = std::chrono::system_clock::now();
	elapsed_seconds = end-start;
	std::cout << "insert time: " << elapsed_seconds.count() << "s\n";
	start = std::chrono::system_clock::now();

	for (int i = 0; i < Max; ++i)
	{
		auto iter = my_tree.find({i, Max - i});
		// to avoid result optimization
		if (!iter->is_valid()) { std::cout << "Error!" << std::endl; }
	}

	end = std::chrono::system_clock::now();
	elapsed_seconds = end-start;
	std::cout << "find time: " << elapsed_seconds.count() << "s\n";
	return 0;
}
