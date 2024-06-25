/*******************************************************************************
 * Functions to aid with endian conversions
 *
 * Author: Levi Miller
 * Date Created: 10/8/2020
 ******************************************************************************/

#ifndef _ENDIAN_H_
#define _ENDIAN_H_

// Returns true if system is Big Endian
static bool SysBigEndian() {
	int x = 1;
	return *(char *)&x != 1;
}

// Swaps two bytes
static inline void _swap(char* const a, char* const b) {
	char t = *a;

	*a = *b;
	*b = t;
}

// Reverses the endianness of input
template<typename T>
static inline void ReverseEndian(T* input) {
	constexpr auto size = sizeof(T);
    char* const data = (char*)input;

    for (int i = 0; i < size / 2; ++i)
		_swap(data + i, data + size - (i + 1));
}

// Reverse the endianness of each element of an array
template<typename T>
static inline void ReverseArrayEndian(T* input, unsigned size) {
	const auto end = input + size;

	for (T* pos = input; pos < end; ++pos)
		ReverseEndian(pos);
}

#endif