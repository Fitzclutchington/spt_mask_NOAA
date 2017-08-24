#ifndef MATRIX_H_
#define MATRIX_H_

#include <string>
#include <ostream>
#include <fstream>
#include <memory>
#include <utility>
#include <cmath>
#include <exception>

////////////////////////////////////////////////////////////////////////////////
//
// This is a lightweight 2-D array (matrix) class for storing data of any
// type. Everything except the base class is optimized for numeric data.
// Assignment is a constant time operation since the storage is aliased
// and reference counted. The data can be cloned if needed, and the matrix can
// wrap a given pointer without taking ownership. Storage is only allocated
// if explicitly requested.
//
////////////////////////////////////////////////////////////////////////////////

// Base class definitions.
#include "matrix_base.h"

// Arithmetic operators.
#include "matrix_oper.h"

// Bitwise operators.
#include "matrix_bits.h"

// Logical operators.
#include "matrix_logi.h"

// Comparison operators.
#include "matrix_comp.h"

// Mathematical functions.
#include "matrix_math.h"

// Statistical functions.
#include "matrix_stat.h"

#endif
