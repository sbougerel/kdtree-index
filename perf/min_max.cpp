#include <iostream>
#include <utility>
#include <chrono>
#include <cstring>

#include "../include/kdtree_index.hpp"

using namespace kdtree_index;

struct pod { int a; };
struct ac_pod
{
	bool operator()(dimension_type, const pod& a, const pod& b) const noexcept
	{ return  a.a < b.a; }
};
typedef indexable<pod, 1, ac_pod> my_indexable;


int main (int, char **, char **)
{
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> elapsed_seconds;

	constexpr int Max = 100000;
	kdtree<my_indexable> my_tree(Max);

	start = std::chrono::system_clock::now();

	for (int i = 0; i < Max; ++i) my_tree.insert({i});

	end = std::chrono::system_clock::now();
	elapsed_seconds = end-start;
	std::cout << "insert time: " << elapsed_seconds.count() << "s\n";
	start = std::chrono::system_clock::now();

	for (int i = 0; i < Max; ++i)
	{
		auto dist = my_tree.end() - my_tree.begin();
		auto offset = root_offset(dist);
		auto iter = root(my_tree.begin(), dist);
		auto max = maximum(0, 0, offset, iter, my_tree.get_index());
		auto min = minimum(0, 0, offset, iter, my_tree.get_index());
		// to avoid result optimization
		if (max == min) { std::cout << "Error!" << std::endl; }
	}

	end = std::chrono::system_clock::now();
	elapsed_seconds = end-start;
	std::cout << "min-max time: " << elapsed_seconds.count() << "s\n";
	return 0;
}
