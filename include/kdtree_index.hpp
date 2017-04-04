#ifndef KDTREE_INDEX_HPP
#define KDTREE_INDEX_HPP

#include <type_traits>
#include <memory>
#include <iterator>
#include <algorithm>
#include <cassert>
#include <cstring>
#include "details/bitwise.hpp"

namespace kdtree_index
{
	typedef std::size_t dimension_type;

	template<dimension_type K>
	dimension_type inc(dimension_type d) noexcept { return (d + 1) % K; }

	struct null_type { };

	template<typename Value,
	         dimension_type K,
	         typename AccessCompare = null_type,
	         typename Accessor = null_type,
	         typename Compare = null_type>
	class indexable
		: private AccessCompare, Accessor, Compare
	{
	public:
		typedef Value value_type;
		typedef AccessCompare access_compare_type;
		typedef Accessor accessor_type;
		typedef Compare compare_type;

		explicit indexable(AccessCompare a = AccessCompare(),
		                   Accessor b = Accessor(),
		                   Compare s = Compare()) noexcept
			: AccessCompare(a), Accessor(b), Compare(s) { }

		const access_compare_type& access_compare() const noexcept
		{ return static_cast<const access_compare_type&>(*this); }

		const accessor_type& accessor() const noexcept
		{ return static_cast<const accessor_type&>(*this); }

		const compare_type& compare() const noexcept
		{ return static_cast<const compare_type&>(*this); }

		static constexpr dimension_type kth() { return K; }
	};

	template<typename Value,
	         dimension_type K,
	         typename Accessor,
	         typename Compare>
	class indexable<Value, K, null_type, Accessor, Compare>
		: private Accessor, Compare
	{
		static_assert(std::is_same<Accessor, null_type>::value,
		              "Accessor is null_type");
		static_assert(std::is_same<Compare, null_type>::value,
		              "Compare is null_type");

	public:
		typedef Value value_type;
		typedef null_type access_compare_type;
		typedef Accessor accessor_type;
		typedef Compare compare_type;

		explicit indexable(Accessor b = Accessor(),
		                   Compare s = Compare()) noexcept
			: Accessor(b), Compare(s) { }

		const accessor_type& accessor() const noexcept
		{ return static_cast<const accessor_type&>(*this); }

		const compare_type& compare() const noexcept
		{ return static_cast<const compare_type&>(*this); }

		static constexpr dimension_type kth() { return K; }
	};

	template<typename Value,
	         dimension_type K,
	         typename AccessCompare>
	class indexable<Value, K, AccessCompare, null_type, null_type>
		: private AccessCompare
	{
	public:
		typedef Value value_type;
		typedef AccessCompare access_compare_type;
		typedef null_type accessor_type;
		typedef null_type compare_type;

		explicit indexable(AccessCompare a = AccessCompare()) noexcept
			: AccessCompare(a) { }

		const access_compare_type& access_compare() const noexcept
		{ return static_cast<const access_compare_type&>(*this); }

		static constexpr dimension_type kth() { return K; }
	};

	template<typename Indexable>
	inline
	typename std::enable_if
	<!std::is_same<typename Indexable::access_compare_type,
	               null_type>::value, bool>::type
	select_compare(dimension_type d,
	               const typename Indexable::value_type& a,
	               const typename Indexable::value_type& b,
	               const Indexable& i) noexcept
	{ return i.access_compare()(d, a, b); }

	template<typename Indexable>
	inline
	typename std::enable_if
	<std::is_same<typename Indexable::access_compare_type,
	              null_type>::value, bool>::type
	select_compare(dimension_type d,
	               const typename Indexable::value_type& a,
	               const typename Indexable::value_type& b,
	               const Indexable& i) noexcept
	{ return i.compare()(i.accessor()(d, a), i.accessor()(d, b)); }

	/**
	 *  State is based on unsigned char; the smallest directly addressable
	 *  type. This leads to good balance between waste of memory (6 bits per
	 *  State) and performance (fast dereferencing).
	 */
	enum class State : unsigned char
	{ Invalid = 0x0, Heads = 0x1, Tails = 0x2, Unsure = 0x3};

	/**
	 *  Swap Invalid for Neutral or Tail for Heads and vice-versa.
	 */
	inline State operator~ (State s) noexcept
	{ return (State)((unsigned char)(s) ^ 0x3); }

	/**
	 *  Combine the values of a and b such that if both values are the same the
	 *  result is unchanged, but if both are different the result is unsure.
	 */
	inline State operator+ (State a, State b) noexcept
	{ return (a == b) ? a : State::Unsure; }

	template<typename ValuePtr, typename StatePtr>
	class kdtree_iterator;

	/**
	 *  A proxy object for kdtree iterator, with a pointer to the value and its
	 *  corresponding state. Before accessing \ref value(), the object should be
	 *  verified with \ref is_valid(). If \ref is_valid() returns false, \ref
	 *  value() is undefined.
	 */
	template<typename ValuePtr, typename StatePtr>
	struct kdtree_element
	{
		using value_pointer = ValuePtr;
		using state_pointer = StatePtr;
		using value_type
		= typename std::pointer_traits<ValuePtr>::element_type;
		using state_type
		= typename std::pointer_traits<StatePtr>::element_type;
		using value_reference = value_type&;
		using state_reference = state_type&;

		explicit kdtree_element() noexcept : _value_ptr(), _state_ptr() { }
		explicit kdtree_element(value_pointer v, state_pointer s) noexcept
			: _value_ptr(v), _state_ptr(s) { }

		bool is_valid() const noexcept { return (*_state_ptr != State::Invalid); }

		state_pointer state_ptr() const noexcept { return _state_ptr; }
		value_pointer value_ptr() const noexcept { return _value_ptr; }

		state_reference state() const noexcept { return *_state_ptr; }
		/**
		 *  Check \ref is_valid() before using this function. If is_valid() returns
		 *  false, \ref value() is undefined.
		 */
		value_reference value() const noexcept { return *_value_ptr; }

		bool operator==(const kdtree_element& other) const noexcept
		{ return _value_ptr == other._value_ptr; }
		bool operator!=(const kdtree_element& other) const noexcept
		{ return !operator==(other); }

	private:
		friend class kdtree_iterator<ValuePtr, StatePtr>;
		value_pointer _value_ptr;
		state_pointer _state_ptr;
	};

	template<typename ValuePtr, typename StatePtr>
	class kdtree_iterator
	{
		typedef std::iterator_traits<ValuePtr> traits_type;

	public:
		typedef kdtree_element<ValuePtr, StatePtr> value_type;
		typedef const kdtree_element<ValuePtr, StatePtr>* pointer;
		typedef const kdtree_element<ValuePtr, StatePtr>& reference;
		typedef typename traits_type::difference_type difference_type;
		typedef std::forward_iterator_tag iterator_category;
		using value_pointer = typename value_type::value_pointer;
		using state_pointer = typename value_type::state_pointer;

		explicit kdtree_iterator() noexcept : _elem() { }
		explicit kdtree_iterator(value_pointer v, state_pointer s) noexcept
			: _elem(v, s) { }

		kdtree_iterator(const kdtree_iterator& x) noexcept
			: _elem(x._elem) { }

		/**
		 *  Converts from iterator<ValuePtr> to iterator<const ValuePtr>. Use SFINAE
		 *  to discard copy constructor overload.
		 */
		template<typename OtherPtr>
		kdtree_iterator(const kdtree_iterator<OtherPtr,
		                typename std::remove_const
		                <typename std::remove_pointer
		                <typename std::enable_if
		                <std::is_same
		                <const typename std::remove_pointer<OtherPtr>::type*,
		                ValuePtr>::value, StatePtr>::type>::type>::type*>& x)
			noexcept
			: _elem(x->value_ptr(), x->state_ptr()) { }

		kdtree_iterator operator++() noexcept
		{
			++_elem._value_ptr; ++_elem._state_ptr;
			return *this;
		}
		kdtree_iterator operator++(int) noexcept
		{
			kdtree_iterator tmp(*this);
			++_elem._value_ptr; ++_elem._state_ptr;
			return tmp;
		}
		kdtree_iterator operator+=(difference_type d) noexcept
		{
			_elem._value_ptr += d; _elem._state_ptr += d;
			return *this;
		}
		kdtree_iterator operator+(difference_type d) const noexcept
		{ return kdtree_iterator(_elem._value_ptr + d, _elem._state_ptr + d); }

		kdtree_iterator operator--() noexcept
		{
			--_elem._value_ptr; --_elem._state_ptr;
			return *this;
		}
		kdtree_iterator operator--(int) noexcept
		{
			kdtree_iterator tmp(*this);
			--_elem._value_ptr; --_elem._state_ptr;
			return tmp;
		}
		kdtree_iterator operator-=(difference_type d) noexcept
		{
			_elem._value_ptr -= d; _elem._state_ptr -= d;
			return *this;
		}
		kdtree_iterator operator-(difference_type d) const noexcept
		{ return kdtree_iterator(_elem._value_ptr - d, _elem._state_ptr - d); }

		reference operator*() const noexcept { return _elem; };
		pointer operator->() const noexcept { return &_elem; };

		kdtree_iterator reset(value_pointer v, state_pointer s) noexcept
		{
			_elem._value_ptr = v;
			_elem._state_ptr = s;
			return *this;
		}

	private:
		value_type _elem;
	};

	template<typename ValuePtr, typename StatePtr>
	inline
	bool operator==(const kdtree_iterator<ValuePtr, StatePtr>& a,
	                const kdtree_iterator<ValuePtr, StatePtr>& b) noexcept
	{ return *a == *b; }

	template<typename ValuePtr, typename StatePtr>
	inline
	bool operator!=(const kdtree_iterator<ValuePtr, StatePtr>& a,
	                const kdtree_iterator<ValuePtr, StatePtr>& b) noexcept
	{ return *a != *b; }

	template<typename ValuePtr, typename StatePtr>
	inline typename kdtree_iterator<ValuePtr, StatePtr>::difference_type
	operator-(const kdtree_iterator<ValuePtr, StatePtr>& a,
	          const kdtree_iterator<ValuePtr, StatePtr>& b) noexcept
	{ return a->value_ptr() - b->value_ptr(); }

	template<typename ValuePtr, typename StatePtr>
	inline kdtree_iterator<ValuePtr, StatePtr>
	left(const kdtree_iterator<ValuePtr, StatePtr>& x,
	     typename kdtree_iterator<ValuePtr, StatePtr>::difference_type o)
		noexcept
	{ return x - o; }

	template<typename ValuePtr, typename StatePtr>
	inline kdtree_iterator<ValuePtr, StatePtr>
	right(const kdtree_iterator<ValuePtr, StatePtr>& x,
	      typename kdtree_iterator<ValuePtr, StatePtr>::difference_type o)
		noexcept
	{ return x + o; }

	template<typename ValuePtr, typename StatePtr>
	inline kdtree_iterator<ValuePtr, StatePtr>
	root(const kdtree_iterator<ValuePtr, StatePtr>& x,
	     typename kdtree_iterator<ValuePtr, StatePtr>::difference_type d)
		noexcept
	{ return x + (d / 2); }

	template<typename Difference>
	inline Difference root_offset(Difference d) noexcept
	{ return (d + 1) / 4; }

	/**
	 *  find the sub-tree with the maximum root value along dimension k.
	 */
	template<typename ValuePtr, typename StatePtr, typename Indexable>
	inline kdtree_iterator<ValuePtr, StatePtr>
	minimum(dimension_type fixed_dim, dimension_type node_dim,
	        typename kdtree_iterator<ValuePtr, StatePtr>
	        ::difference_type node_offset,
	        kdtree_iterator<ValuePtr, StatePtr> node,
	        const Indexable& index) noexcept
	{
		using iterator = kdtree_iterator<ValuePtr, StatePtr>;
		iterator best = node;
		while (node_offset > 1)
		{
			iterator child;
			dimension_type child_dim = inc<Indexable::kth()>(node_dim);
			typename iterator::difference_type child_offset = node_offset / 2;
			child = minimum(fixed_dim, child_dim, child_offset,
			                left(node, node_offset), index);
			if (!select_compare(fixed_dim, best->value(), child->value(), index))
			{ best = child; }
			if (node_dim == fixed_dim)
			{ return best; }
			child = right(node, node_offset);
			if (!select_compare(fixed_dim, best->value(), child->value(), index))
			{ best = child; }
			node = child;
			node_dim = child_dim;
			node_offset = child_offset;
		}
		if (node_offset == 1)
		{
			iterator child = left(node, node_offset);
			if (child->is_valid()
			    && !select_compare(fixed_dim, best->value(), child->value(), index))
			{ best = child; }
			child = right(node, node_offset);
			if (child->is_valid()
			    && !select_compare(fixed_dim, best->value(), child->value(), index))
			{ best = child; }
		}
		return best;
	}

	/**
	 *  find the sub-tree with the maximum root value along dimension k.
	 */
	template<typename ValuePtr, typename StatePtr, typename Indexable>
	inline kdtree_iterator<ValuePtr, StatePtr>
	maximum(dimension_type fixed_dim, dimension_type node_dim,
	        typename kdtree_iterator<ValuePtr, StatePtr>
	        ::difference_type node_offset,
	        kdtree_iterator<ValuePtr, StatePtr> node,
	        const Indexable& index) noexcept
	{
		using iterator = kdtree_iterator<ValuePtr, StatePtr>;
		iterator best = node;
		while (node_offset > 1)
		{
			iterator child;
			dimension_type child_dim = inc<Indexable::kth()>(node_dim);
			typename iterator::difference_type child_offset = node_offset / 2;
			child = maximum(fixed_dim, child_dim, child_offset,
			                right(node, node_offset), index);
			if (!select_compare(fixed_dim, child->value(), best->value(), index))
			{ best = child; }
			if (node_dim == fixed_dim)
			{ return best; }
			child = left(node, node_offset);
			if (!select_compare(fixed_dim, child->value(), best->value(), index))
			{ best = child; }
			node = child;
			node_dim = child_dim;
			node_offset = child_offset;
		}
		if (node_offset == 1)
		{
			iterator child = left(node, node_offset);
			if (child->is_valid()
			    && !select_compare(fixed_dim, child->value(), best->value(), index))
			{ best = child; }
			child = right(node, node_offset);
			if (child->is_valid()
			    && !select_compare(fixed_dim, child->value(), best->value(), index))
			{ best = child; }
		}
		return best;
	}

	template<typename Index,
	         typename Alloc = std::allocator<typename Index::value_type>>
	class kdtree
	{
	public:
		using indexable_type = Index;
		using value_type = typename indexable_type::value_type;
		using state_type = State;
		using allocator_type = Alloc;

	private:
		using value_alloc_type = typename std::allocator_traits<Alloc>
			::template rebind_alloc<value_type>;
		using state_alloc_type = typename std::allocator_traits<Alloc>
			::template rebind_alloc<state_type>;
		using value_alloc_traits = std::allocator_traits<value_alloc_type>;
		using state_alloc_traits = std::allocator_traits<state_alloc_type>;
		using value_pointer = typename value_alloc_traits::pointer;
		using const_value_pointer = typename value_alloc_traits::const_pointer;
		using state_pointer = typename state_alloc_traits::pointer;
		using const_state_pointer = typename state_alloc_traits::const_pointer;

	public:
		using iterator = kdtree_iterator<value_pointer, state_pointer>;
		using const_iterator = kdtree_iterator<const_value_pointer, const_state_pointer>;

	private:
		struct _kdtree_members
			: indexable_type, value_alloc_type, state_alloc_type
		{
			iterator _start;          // first of value & state
			iterator _finish;         // last of value & state
			std::size_t _capacity;    // total storage capacity
			std::size_t _count;       // fast count for O(1) access
			state_type _full_state;   // State indicating a perfectly balanced tree

			explicit _kdtree_members()
			noexcept(std::is_nothrow_default_constructible<value_alloc_type>::value
			         && std::is_nothrow_default_constructible<state_alloc_type>::value)
			: indexable_type(), value_alloc_type(), state_alloc_type(),
			  _start(), _finish(_start), _capacity(), _count(),
			  _full_state(State::Heads) { }

			explicit _kdtree_members(const indexable_type& i,
			                         const value_alloc_type& a,
			                         const state_alloc_type& s)
				noexcept(std::is_nothrow_copy_constructible<value_alloc_type>::value
				         && std::is_nothrow_copy_constructible<state_alloc_type>::value)
				: indexable_type(i), value_alloc_type(a), state_alloc_type(s),
				  _start(), _finish(_start), _capacity(), _count(),
				  _full_state(State::Heads) { }

			_kdtree_members(const _kdtree_members& x)
				noexcept(std::is_nothrow_copy_constructible<value_alloc_type>::value
				         && std::is_nothrow_copy_constructible<state_alloc_type>::value)
				: indexable_type(static_cast<const indexable_type&>(x)),
				  value_alloc_type(static_cast<const value_alloc_type&>(x)),
				  state_alloc_type(static_cast<const state_alloc_type&>(x)),
				  _start(), _finish(_start), _capacity(), _count(),
				  _full_state(x._full_state) { }

			_kdtree_members(_kdtree_members&& x)
				noexcept(std::is_nothrow_move_constructible<value_alloc_type>::value
				         && std::is_nothrow_move_constructible<state_alloc_type>::value)
				: indexable_type(static_cast<indexable_type&&>(x)),
				  value_alloc_type(static_cast<value_alloc_type&&>(x)),
				  state_alloc_type(static_cast<state_alloc_type&&>(x)),
				  _start(), _finish(_start), _capacity(), _count(),
				  _full_state(State::Heads)
			{
				std::swap(_start, x._start);
				std::swap(_finish, x._finish);
				std::swap(_capacity, x._capacity);
				std::swap(_count, x._count);
				std::swap(_full_state, x._full_state);
			}
		} _impl;

		value_alloc_type& _get_value_alloc() noexcept
		{ return static_cast<value_alloc_type&>(_impl); }
		const value_alloc_type& _get_value_alloc() const noexcept
		{ return static_cast<value_alloc_type&>(_impl); }
		state_alloc_type& _get_state_alloc() noexcept
		{ return static_cast<state_alloc_type&>(_impl); }
		const state_alloc_type& _get_state_alloc() const noexcept
		{ return static_cast<state_alloc_type&>(_impl); }

		/**
		 *  Create initial storage for the flat tree. Always allocate the smallest
		 *  power of 2 that fits the requested size so that a perfectly balanced
		 *  tree fits in.
		 */
		void _alloc_storage(std::size_t n)
		{
			if (n == 0) { return; }
			n = details::bitwise<std::size_t>::ftz(n);
			value_pointer v = value_alloc_traits::allocate(_get_value_alloc(), n);
			state_pointer s;
			try
			{ s = state_alloc_traits::allocate(_get_state_alloc(), n); }
			catch(...)
			{
				value_alloc_traits::deallocate(_get_value_alloc(), v, n);
				throw;
			}
			_impl._capacity = n;
			_impl._start.reset(v, s);
			_impl._finish = _impl._start;
		}

		void _dealloc_storage()
		{
			value_alloc_traits::deallocate
				(_get_value_alloc(), _impl._start->value_ptr(), _impl._capacity);
			state_alloc_traits::deallocate
				(_get_state_alloc(), _impl._start->state_ptr(), _impl._capacity);
			_impl._capacity = 0;
		}

		/**
		 *  Expand the tree by inserting a Invalid value between each exisiting values. Can
		 *  expand with overlapping memory segments.
		 */
		void _expand(value_pointer vp, state_pointer cp) noexcept
		{
			auto offset = (_impl._finish - _impl._start) * 2;
			iterator slow(_impl._finish);
			iterator first(vp, cp);
			iterator last(vp + offset, cp + offset);
			last->state() = State::Invalid;
			while (last != first)
			{
				--last; --slow;
				last->state() = slow->state();
				std::memcpy(last->value_ptr(), slow->value_ptr(), sizeof(value_type));
				--last;
				last->state() = State::Invalid;
			}
			_impl._finish.reset(vp + offset + 1, cp + offset + 1);
		}

		void _expand_alloc()
		{
			std::size_t n = (_impl._capacity * 2) + 1;
			value_pointer vp = value_alloc_traits::allocate(_get_value_alloc(), n);
			state_pointer cp;
			try
			{ cp = state_alloc_traits::allocate(_get_state_alloc(), n); }
			catch (...)
			{
				value_alloc_traits::deallocate(_get_value_alloc(), vp, n);
				throw;
			}
			_expand(vp, cp);
			value_alloc_traits::deallocate(_get_value_alloc(),
			                               _impl._start->value_ptr(), _impl._capacity);
			state_alloc_traits::deallocate(_get_state_alloc(),
			                               _impl._start->state_ptr(), _impl._capacity);
			_impl._start.reset(vp, cp);
			_impl._capacity = n;
		}

		void _prepare_insert()
		{
			if (_impl._count == 0)
			{
				if (_impl._capacity == 0)
				{ _alloc_storage(1); }
				_impl._finish = _impl._start + 1;
			}
			else
				if (_impl._count == static_cast<std::size_t>(_impl._finish - _impl._start))
				{
					if (_impl._count == _impl._capacity) { _expand_alloc(); }
					else { _expand(_impl._start->value_ptr(), _impl._start->state_ptr()); }
					_impl._full_state = ~_impl._full_state;
				}
		}

		/**
		 *  Collapse the tree by removing all leaves of the tree. The leaves' parent
		 *  will become the new leaves. This operation corresponds exactly to
		 *  removing one item between each item, collapsing the tree to half of its
		 *  size.
		 */
		void _collapse() noexcept
		{
			auto offset = (_impl.finish() - _impl.start()) / 2;
			iterator fast(_impl._start->value_ptr(), _impl._start->state_ptr());
			iterator first(fast);
			iterator last(_impl._start->value_ptr() + offset,
			              _impl._start->state_ptr() + offset);
			++fast; // skip first;
			while (last != first)
			{
				std::memcpy(first->value_ptr(), fast->value_ptr(), sizeof(value_type));
				first->state() = fast->state();
				++first;
				fast += 2; // skip invalid states;
			}
			_impl._finish.reset(_impl._start->value_ptr() + offset,
			                    _impl._start->state_ptr() + offset);
		}

		/**
		 *  Destroy all items in the flat tree, calling the destructor for each
		 *  value_type.
		 *
		 *  @note If a call to ~value_type() throws an exception, the behavior is
		 *  undefined, as the tree is left partially initialized and most function
		 *  may not work.
		 */
		void _destroy()
			noexcept(std::is_nothrow_destructible<value_type>::value)
		{
			for (iterator i = _impl._start; i != _impl._finish; ++i)
			{ if (i->is_valid()) { i->value_ptr()->~value_type(); } }
			_impl._finish = _impl._start;
			_impl._count = 0;
		}

		struct _deferred_memcpy
		{
			explicit _deferred_memcpy(value_pointer x) noexcept : p(x) { }
			void place_at(value_pointer x) const noexcept
			{ std::memcpy(x, p, sizeof(value_type)); }
			const value_type& cref() const noexcept { return *p; }
			value_pointer p;
		};

		struct _deferred_copy
		{
			explicit _deferred_copy(const value_type& x) noexcept : p(x) { }
			void place_at(value_pointer x) const
				noexcept(std::is_nothrow_copy_constructible<value_type>::value)
			{ new(x) value_type(p); }
			const value_type& cref() const noexcept { return p; }
			const value_type& p;
		};

		struct _deferred_move
		{
			explicit _deferred_move(value_type&& x) noexcept : p(std::move(x)) { }
			void place_at(value_pointer x) const
				noexcept(std::is_nothrow_move_constructible<value_type>::value)
			{ new(x) value_type(std::move(p)); }
			const value_type& cref() const noexcept { return p; }
			value_type&& p;
		};

		/**
		 *  List initialization works with the average O(n.log(n)) algorithm. This
		 *  algorithm may have worst case performance of O(n^2).
		 */
		void _uninitialized_insert
		(typename std::initializer_list<value_type>::iterator,
		 typename std::initializer_list<value_type>::iterator)
			noexcept(std::is_nothrow_copy_constructible<value_type>::value)
		{
		}

		/**
		 *  This function is to be called only when node is full. It will not
		 *  proceed to actual detruction of "erased", so it is important that the
		 *  caller destroys "erased" before calling the function.
		 */
		void _erase_when_full(dimension_type node_dim,
		                      typename iterator::difference_type node_offset,
		                      const iterator& node,
		                      const iterator& erased) const noexcept
		{
			if (node_offset > 1)
			{
				dimension_type child_dim = inc<indexable_type::kth()>(node_dim);
				typename iterator::difference_type child_offset = node_offset / 2;
				iterator lnode = left(node, node_offset);
				iterator rnode = right(node, node_offset);
				if (node == erased)
				{
					iterator tmp = minimum(node_dim, child_dim, child_offset,
					                       rnode, get_index());
					std::memcpy(erased->value_ptr(), tmp->value_ptr(),
					            sizeof(value_type));
					_erase_when_full(child_dim, child_offset, rnode, tmp);
				}
				// find erased node by memory locality
				else if (node->value_ptr() < erased->value_ptr())
				{ _erase_when_full(child_dim, child_offset, rnode, erased); }
				else
				{ _erase_when_full(child_dim, child_offset, lnode, erased); }
				// Fix states before returning
				node->state() = State::Unsure;
			}
			else if (node_offset == 1)
			{
				iterator rnode = right(node, node_offset);
				if (node == erased)
				{
					std::memcpy(node->value_ptr(), rnode->value_ptr(),
					            sizeof(value_type));
					rnode->state() = State::Invalid;
				}
				else { erased->state() = State::Invalid; }
				node->state() = State::Unsure;
			}
			else
			{ node->state() = State::Invalid; }
			return;
		}

		/**
		 *  Find a position to insert v into the subtree (start, root) and balance
		 *  all items along the way to make way for v.
		 */
		template<typename DeferredOp>
		iterator _single_insert(dimension_type node_dim,
		                         typename iterator::difference_type offset,
		                         const iterator& node,
		                         const DeferredOp& defer) const
			noexcept(noexcept(defer.place_at(node->value_ptr())))
		{
			if (offset == 1)
			{
				iterator lnode = left(node, offset);
				iterator rnode = right(node, offset);
				iterator insert;
				if (select_compare(node_dim, defer.cref(), node->value(), get_index()))
				{
					if (lnode->is_valid())
					{
						std::memcpy(rnode->value_ptr(), node->value_ptr(), sizeof(value_type));
						rnode->state() = _impl._full_state;
						node->state() = _impl._full_state;
						if (select_compare(node_dim, defer.cref(), lnode->value(), get_index()))
						{
							std::memcpy(node->value_ptr(), lnode->value_ptr(), sizeof(value_type));
							defer.place_at(lnode->value_ptr());
							insert = lnode;
						}
						else
						{
							defer.place_at(node->value_ptr());
							insert = node;
						}
					}
					else
					{
						defer.place_at(lnode->value_ptr());
						lnode->state() = _impl._full_state;
						if (rnode->is_valid()) { node->state() = _impl._full_state; }
						insert = lnode;
					}
				}
				else
				{
					if (rnode->is_valid())
					{
						std::memcpy(lnode->value_ptr(), node->value_ptr(), sizeof(value_type));
						lnode->state() = _impl._full_state;
						node->state() = _impl._full_state;
						if (select_compare(node_dim, rnode->value(), defer.cref(), get_index()))
						{
							std::memcpy(node->value_ptr(), rnode->value_ptr(), sizeof(value_type));
							defer.place_at(rnode->value_ptr());
							insert = rnode;
						}
						else
						{
							defer.place_at(node->value_ptr());
							insert = node;
						}
					}
					else
					{
						defer.place_at(rnode->value_ptr());
						rnode->state() = _impl._full_state;
						if (lnode->is_valid()) { node->state() = _impl._full_state; }
						insert = rnode;
					}
				}
				return insert;
			}
			else if (offset > 1)
			{
				dimension_type child_dim = inc<indexable_type::kth()>(node_dim);
				typename iterator::difference_type child_offset = offset / 2;
				iterator lnode = left(node, offset);
				iterator rnode = right(node, offset);
				iterator insert;
				if (select_compare(node_dim, defer.cref(), node->value(), get_index()))
				{
					if (lnode->state() == _impl._full_state)
					{
						_single_insert(child_dim, child_offset, rnode,
						               _deferred_memcpy(node->value_ptr()));
						iterator tmp = maximum(node_dim, child_dim, child_offset, lnode,
						                       get_index());
						if (select_compare(node_dim, defer.cref(), tmp->value(), get_index()))
						{
							std::memcpy(node->value_ptr(), tmp->value_ptr(), sizeof(value_type));
							_erase_when_full(child_dim, child_offset, lnode, tmp);
							insert = _single_insert(child_dim, child_offset, lnode, defer);
						}
						else
						{
							defer.place_at(node->value_ptr());
							insert = node;
						}
					}
					else
					{ insert = _single_insert(child_dim, child_offset, lnode, defer); }
				}
				else if (select_compare(node_dim, node->value(), defer.cref(), get_index()))
				{
					if (rnode->state() == _impl._full_state)
					{
						_single_insert(child_dim, child_offset, lnode,
						               _deferred_memcpy(node->value_ptr()));
						iterator tmp = minimum(node_dim, child_dim, child_offset, rnode,
						                       get_index());
						if (select_compare(node_dim, tmp->value(), defer.cref(), get_index()))
						{
							std::memcpy(node->value_ptr(), tmp->value_ptr(), sizeof(value_type));
							_erase_when_full(child_dim, child_offset, rnode, tmp);
							insert = _single_insert(child_dim, child_offset, rnode, defer);
						}
						else
						{
							defer.place_at(node->value_ptr());
							insert = node;
						}
					}
					else
					{ insert = _single_insert(child_dim, child_offset, rnode, defer); }
				}
				else
				{
					insert = (lnode->state() == _impl._full_state)
						? _single_insert(child_dim, child_offset, rnode, defer)
						: _single_insert(child_dim, child_offset, lnode, defer);
				}
				// modify state accordingly and return insert
				node->state() = lnode->state() + rnode->state();
				return insert;
			}
			else
			{
				// offset == 0, necessarily empty
				defer.place_at(node->value_ptr());
				node->state() = _impl._full_state;
				return node;
			}
		}

		iterator
		_find(dimension_type node_dim,
		      typename iterator::difference_type node_offset,
		      iterator node, const value_type& val) const noexcept
		{
			for (; node->is_valid();)
			{
				bool left_only
					= select_compare(node_dim, val, node->value(), get_index());
				bool right_only
					= select_compare(node_dim, node->value(), val, get_index());
				if (!left_only && !right_only)
				{
					dimension_type i = 0;
					for (; i < node_dim; ++i)
					{
						if (select_compare(i, node->value(), val, get_index())
						    || select_compare(i, val, node->value(), get_index()))
						{ break; }
					}
					if (i == node_dim)
					{
						for (++i; i < indexable_type::kth(); ++i)
						{
							if (select_compare(i, node->value(), val, get_index())
							    || select_compare(i, val, node->value(), get_index()))
							{ break; }
						}
						if (i == indexable_type::kth()) { return node; }
					}
				}
				if (node_offset != 0)
				{
					dimension_type child_dim = inc<indexable_type::kth()>(node_dim);
					auto child_offset = node_offset / 2;
					if (!right_only)
					{
						iterator probe = _find(child_dim, child_offset,
						                       left(node, node_offset), val);
						if (probe != _impl._finish) { return probe; }
					}
					if (left_only) { break; }
					node = right(node, node_offset);
					node_dim = child_dim;
					node_offset = child_offset;
				}
				else { break; }
			}
			return _impl._finish;
		}

	public:
		explicit kdtree()
		noexcept(std::is_nothrow_default_constructible<_kdtree_members>::value)
			: _impl() { }

		explicit
		kdtree(const indexable_type& i,
		       const allocator_type& a = allocator_type())
			noexcept(std::is_nothrow_constructible
			         <_kdtree_members,
			         const indexable_type&,
			         const value_alloc_type&,
			         const state_alloc_type&>::value)
			: _impl(i, a, a) { }

		explicit kdtree(std::size_t n)
			noexcept(std::is_nothrow_default_constructible<_kdtree_members>::value)
			: _impl()
		{ _alloc_storage(n); }

		explicit kdtree(std::size_t n,
		                const indexable_type& i,
		                const allocator_type& a = allocator_type())
			noexcept(std::is_nothrow_constructible
			         <_kdtree_members,
			         const indexable_type&,
			         const value_alloc_type&,
			         const state_alloc_type&>::value)
			: _impl(i, a, a)
		{ _alloc_storage(n); }

		kdtree(const kdtree& x)
			: _impl(x._impl)
		{
			_alloc_storage(x._impl._capacity);
			try
			{
				std::uninitialized_copy(x._impl._start->value_ptr(),
				                        x._impl._finish->value_ptr(),
				                        _impl._start->value_ptr());
			}
			catch (...)
			{ _dealloc_storage(); throw; }
			auto dist = x._impl._finish - x._impl._start;
			std::memcpy(_impl._start->state_ptr(), x._impl._start->state_ptr(),
			            static_cast<std::size_t>(dist));
			_impl._finish.reset(_impl._start->value_ptr() + dist,
			                    _impl._start->state_ptr() + dist);
			_impl._count = x._impl._count;
			_impl._full_state = x._impl._full_state;
		}

		kdtree(kdtree&& x)
			noexcept(std::is_nothrow_move_constructible<_kdtree_members>::value)
			: _impl(std::move(x._impl)) { }

		/**
		 *  Insert items in kdtree according to the O(n.log(n)) algorithm: partially
		 *  sort all items in an array to find the root, then recursively process
		 *  each side of the array.
		 */
		kdtree(std::initializer_list<value_type> l,
		       const indexable_type& i = indexable_type(),
		       const allocator_type& a = allocator_type())
			: kdtree(i, a)
		{
			_alloc_storage(l.size());
			_uninitialized_insert(l.begin(), l.end());
		}

		~kdtree()
		{
			if (_impl._capacity != 0)
			{
				clear();
				_dealloc_storage();
			}
		}

		iterator begin() noexcept { return _impl._start; }
		const_iterator begin() const noexcept { return _impl.start; }
		const_iterator cbegin() const noexcept { return _impl.start; }
		iterator end() noexcept { return _impl._finish; }
		const_iterator end() const noexcept { return _impl._finish; }
		const_iterator cend() const noexcept { return _impl._finish; }

		allocator_type get_allocator() const
			noexcept(std::is_nothrow_copy_constructible<allocator_type>::value)
		{ return allocator_type(_get_value_alloc()); }

		const indexable_type& get_index() const noexcept
		{ return static_cast<const indexable_type&>(_impl); }
		indexable_type& get_index() noexcept
		{ return static_cast<indexable_type&>(_impl); }

		std::size_t capacity() const noexcept { return _impl._capacity; }
		std::size_t size() const noexcept { return _impl._count; }
		bool empty() const noexcept { return (size() == 0); }

		void clear()
			noexcept(std::is_nothrow_destructible<value_type>::value)
		{ if (_impl._count != 0) _destroy(); }

		iterator
		insert(const value_type& v)
		{
			_prepare_insert();
			++_impl._count;
			auto dist = _impl._finish - _impl._start;
			return _single_insert(0, root_offset(dist), root(_impl._start, dist),
			                      _deferred_copy(v));
		}

		iterator
		insert(value_type&& v)
		{
			_prepare_insert();
			++_impl._count;
			auto dist = _impl._finish - _impl._start;
			return _single_insert(0, root_offset(dist), root(_impl._start, dist),
			                      _deferred_move(std::move(v)));
		}

		std::size_t erase(const value_type&)
		{
			--_impl._count;
			return 0;
		}

		void erase(iterator)
		{
			--_impl._count;
		}

		iterator
		find(const value_type& val) noexcept
		{
			auto dist = _impl._finish - _impl._start;
			return (_impl._count == 0) ? _impl._finish
				: _find(0, root_offset(dist), root(_impl._start, dist), val);
		}

		const_iterator
		find(const value_type& val) const noexcept
		{
			auto dist = _impl._finish - _impl._start;
			return (_impl._count == 0) ? const_iterator(_impl._finish)
				: _find(0, root_offset(dist), root(_impl._start, dist), val);
		}
	};
}

#endif
