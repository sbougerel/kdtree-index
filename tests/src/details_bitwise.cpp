/**
 *  Uses Boost.Test; needs to be compiled with -lboost_unit_test_framework and
 *  to run it you must have compiled the Boost unit test framework as a shared
 *  library (DLL - for Windows developers). If you do not have the shared
 *  library at your disposal, you can remove all occurrences of
 *  BOOST_TEST_DYN_LINK, and the library will be linked with the static
 *  library. This still requires that you have compiled the Boost unit test
 *  framework as a static library.
 */

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "../../include/details/bitwise.hpp"
using namespace kdtree_index::details;

BOOST_AUTO_TEST_CASE(uint32_msb)
{
	BOOST_CHECK_EQUAL(1, bitwise<std::uint32_t>::ftz(1));
	BOOST_CHECK_EQUAL(0xF, bitwise<std::uint32_t>::ftz(0x8));
	BOOST_CHECK_EQUAL(0xFFFFFFFF, bitwise<std::uint32_t>::ftz(0xFFFFFFFF));
	BOOST_CHECK_EQUAL(0xFFFFFFFF, bitwise<std::uint32_t>::ftz(0x80000000));
}

BOOST_AUTO_TEST_CASE(uint64_msb)
{
	BOOST_CHECK_EQUAL(1, bitwise<std::uint64_t>::ftz(1));
	BOOST_CHECK_EQUAL(0xF, bitwise<std::uint64_t>::ftz(0x8));
	BOOST_CHECK_EQUAL(0xFFFFFFFFFFFFFFFF, bitwise<std::uint64_t>::ftz(0xFFFFFFFFFFFFFFFF));
	BOOST_CHECK_EQUAL(0xFFFFFFFFFFFFFFFF, bitwise<std::uint64_t>::ftz(0x8000000000000000));
}
