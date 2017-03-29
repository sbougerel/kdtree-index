#ifndef BITWISE_HPP
#define BITWISE_HPP

namespace kdtree_index
{
	namespace details
	{
		template<typename U, int = sizeof(U)>
		struct bitwise { };

		template<typename U>
		struct bitwise<U, 2>
		{
			// Fill all trailing zeroes after the leading 1
			static U ftz(U u) noexcept
			{
				u |= u >> 1;
				u |= u >> 2;
				u |= u >> 4;
				u |= u >> 8;
				return u;
			}
		};

		template<typename U>
		struct bitwise<U, 4>
		{
			// Fill all trailing zeroes after the leading 1
			static U ftz(U u) noexcept
			{
				u |= u >> 1;
				u |= u >> 2;
				u |= u >> 4;
				u |= u >> 8;
				u |= u >> 16;
				return u;
			}
		};

		template<typename U>
		struct bitwise<U, 8>
		{
			// Fill all trailing zeroes after the leading 1
			static U ftz(U u) noexcept
			{
				u |= u >> 1;
				u |= u >> 2;
				u |= u >> 4;
				u |= u >> 8;
				u |= u >> 16;
				u |= u >> 32;
				return u;
			}
		};
	}
}

#endif
