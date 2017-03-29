/**
 *  Uses Boost.Test; needs to be compiled with -lboost_unit_test_framework and
 *  to run it you must have compiled the Boost unit test framework as a shared
 *  library (DLL - for Windows developers). If you do not have the shared
 *  library at your disposal, you can remove all occurrences of
 *  BOOST_TEST_DYN_LINK, and the library will be linked with the static
 *  library. This still requires that you have compiled the Boost unit test
 *  framework as a static library.
 */

#include <cstdlib> // std::rand(), std::srand()
#include <ctime>
#include <cstddef>
#include <iostream>

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(install_srand)
{
  // Not sure where else to put this...
  std::srand((unsigned int)std::time(NULL));
}

#include "../../include/kdtree_index.hpp"
using namespace kdtree_index;

struct pod { int a; };
struct ac_pod
{
	bool operator()(dimension_type, const pod& a, const pod& b) const noexcept
	{ return  a.a < b.a; }
};
typedef indexable<pod, 1, ac_pod> my_indexable;

BOOST_AUTO_TEST_CASE(kdtree_default_constructible)
{
	kdtree<my_indexable> my_tree;
	BOOST_CHECK_EQUAL(1, my_tree.get_index().kth());
	BOOST_CHECK_EQUAL(0, my_tree.size());
	BOOST_CHECK_EQUAL(0, my_tree.capacity());
	BOOST_CHECK_EQUAL(true, my_tree.empty());
	BOOST_CHECK(my_tree.begin() == my_tree.end());
}

BOOST_AUTO_TEST_CASE(kdtree_allocate_constructor)
{
	kdtree<my_indexable> my_tree(10);
	BOOST_CHECK_EQUAL(1, my_tree.get_index().kth());
	BOOST_CHECK_EQUAL(0, my_tree.size());
	BOOST_CHECK_EQUAL(true, my_tree.empty());
	BOOST_CHECK(my_tree.begin() == my_tree.end());
	// but this time, 10 fills to 15
	BOOST_CHECK_EQUAL(15, my_tree.capacity());
}

BOOST_AUTO_TEST_CASE(kdtree_copy_constructor)
{
	kdtree<my_indexable> one_tree(10);
	kdtree<my_indexable> two_tree(one_tree);
	BOOST_CHECK_EQUAL(1, two_tree.get_index().kth());
	BOOST_CHECK_EQUAL(0, two_tree.size());
	BOOST_CHECK_EQUAL(true, two_tree.empty());
	BOOST_CHECK(two_tree.begin() == two_tree.end());
	// but this time, 10 fills to 15
	BOOST_CHECK_EQUAL(15, two_tree.capacity());
}

BOOST_AUTO_TEST_CASE(kdtree_move_constructor)
{
	kdtree<my_indexable> one_tree(10);
	BOOST_CHECK_EQUAL(15, one_tree.capacity());
	kdtree<my_indexable> two_tree(std::move(one_tree));
	BOOST_CHECK_EQUAL(15, two_tree.capacity());
	BOOST_CHECK_EQUAL(0, one_tree.capacity());
	BOOST_CHECK(one_tree.begin() == one_tree.end());
	BOOST_CHECK(two_tree.begin() == two_tree.end());
}

BOOST_AUTO_TEST_CASE(kdtree_insert_one_unallocated)
{
	kdtree<my_indexable> tree;
	tree.insert({1});
	BOOST_CHECK_EQUAL(1, tree.capacity());
	BOOST_CHECK_EQUAL(1, tree.size());
	BOOST_CHECK_EQUAL(false, tree.empty());
	BOOST_CHECK(tree.begin() != tree.end());
	BOOST_CHECK(tree.begin()->value().a == 1);
}

BOOST_AUTO_TEST_CASE(kdtree_insert_one_allocated)
{
	kdtree<my_indexable> tree(10);
	tree.insert({1});
	BOOST_CHECK_EQUAL(15, tree.capacity());
	BOOST_CHECK_EQUAL(1, tree.size());
	BOOST_CHECK_EQUAL(false, tree.empty());
	BOOST_CHECK(tree.begin() != tree.end());
	BOOST_CHECK(tree.begin()->value().a == 1);
}

BOOST_AUTO_TEST_CASE(kdtree_insert_ascending)
{
	constexpr int Max = 30;
	kdtree<my_indexable> tree(Max);
	for (int i = 0; i < Max; ++i) tree.insert({i + 1});
	BOOST_CHECK_EQUAL(details::bitwise<int>::ftz(Max), tree.capacity());
	BOOST_CHECK_EQUAL(Max, tree.size());
	BOOST_CHECK_EQUAL(false, tree.empty());
	BOOST_CHECK(tree.begin() != tree.end());
	int check_count = 0;
	int check_seq_val = 0;
	for (auto ref : tree)
	{
		if (ref.is_valid())
		{
			++check_count;
			BOOST_CHECK_LT(check_seq_val, ref.value().a);
			check_seq_val = ref.value().a;
			// std::cout << ref.value().a << std::endl;
		}
		// else std::cout << "Invalid" << std::endl;
	}
	BOOST_CHECK_EQUAL(check_seq_val, Max);
	BOOST_CHECK_EQUAL(check_count, Max);
}

BOOST_AUTO_TEST_CASE(kdtree_insert_descending)
{
	constexpr int Max = 30;
	kdtree<my_indexable> tree(Max);
	for (int i = 0; i < Max; ++i) tree.insert({Max - i});
	BOOST_CHECK_EQUAL(details::bitwise<int>::ftz(Max), tree.capacity());
	BOOST_CHECK_EQUAL(Max, tree.size());
	BOOST_CHECK_EQUAL(false, tree.empty());
	BOOST_CHECK(tree.begin() != tree.end());
	int check_count = 0;
	int check_seq_val = 0;
	for (auto ref : tree)
	{
		if (ref.is_valid())
		{
			++check_count;
			BOOST_CHECK_LT(check_seq_val, ref.value().a);
			check_seq_val = ref.value().a;
			// std::cout << ref.value().a << std::endl;
		}
		// else std::cout << "Invalid" << std::endl;
	}
	BOOST_CHECK_EQUAL(check_seq_val, Max);
	BOOST_CHECK_EQUAL(check_count, Max);
}

BOOST_AUTO_TEST_CASE(kdtree_insert_all_same)
{
	kdtree<my_indexable> tree;
	tree.insert({2});
	tree.insert({2});
	tree.insert({2});
	tree.insert({2});
	tree.insert({2});
	tree.insert({2});
	tree.insert({2});
	tree.insert({2});
	tree.insert({2});
	tree.insert({2});
	tree.insert({2});
	BOOST_CHECK_EQUAL(15, tree.capacity());
	BOOST_CHECK_EQUAL(11, tree.size());
	BOOST_CHECK_EQUAL(false, tree.empty());
	BOOST_CHECK(tree.begin() != tree.end());
	int check_count = 0;
	for (auto ref : tree)
	{
		if (ref.is_valid())
		{
			++check_count;
			BOOST_CHECK_EQUAL(2, ref.value().a);
			// std::cout << ref.value().a << std::endl;
		}
		// else std::cout << "Invalid" << std::endl;
	}
	BOOST_CHECK_EQUAL(check_count, 11);
}
