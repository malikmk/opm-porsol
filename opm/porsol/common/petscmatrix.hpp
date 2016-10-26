#ifndef OPM_PETSCMATRIX_H
#define OPM_PETSCMATRIX_H

#include <opm/core/linalg/petsc.hpp>
#include <opm/core/linalg/petscmixins.hpp>


#include <petscmat.h>

#include <memory>
#include <vector>

namespace Opm {
namespace Petsc {

class Vector;

template<>
struct deleter< _p_Mat >
{ void operator()( Mat x ) { MatDestroy( &x ); } };

/// @brief PETSc Matrix object
///
/// This object is mostly structurally immutable, that is, once it is
/// constructed its nonzero pattern cannot be modified except through
/// Matrix arithmetic-assignment operations. To construct a new Matrix,
/// please see the Matrix::Builder class.
///
/// Matrix is implicitly convertible to Mat, so it can be used directly in
/// PETSc functions. However, I do not recommend this for other things than
/// development and debugging - if some feature is needed from the Vector,
/// consider implementing it in the library.

class Matrix : public uptr< Mat > {
    public:
        typedef PetscScalar scalar;
        typedef PetscInt size_type;

        class Builder;

        enum class nonzero_pattern { different, subset, same };

        Matrix() = delete;

        /// @brief Acquire ownership of a raw PETSc handle
        Matrix( Mat );
        /// @brief Copy constructor.
        Matrix( const Matrix& );
        /// @brief Copy constructor.
        Matrix( Matrix&& );

        /// @brief  Construct a (dense) Matrix from std::vector
        /// Interprets the Vector as a logical rows*columns 2D array.
        /// \param[in] Vector       Logically 2D std::vector to copy from
        /// \param[in] rows         Number of rows
        /// \param[in] columns      Number of columns
        Matrix( const std::vector< scalar >&, size_type, size_type );

        Matrix& operator=( const Matrix& );
        Matrix& operator=( Matrix&& );

        /// @brief Get number of rows.
        /// \param[out] size    Number of rows in the Matrix
        size_type rows() const;
        /// @brief Get number of cols.
        /// \param[out] size    Number of columns in the Matrix
        size_type cols() const;

        /// @brief Scalar multiplication, A = cA.
        /// \param[in] scalar   Value to scale with
        Matrix& operator*=( scalar );
        /// @brief Inverse scalar multiplication (division)
        /// \param[in] scalar   Value to scale with
        Matrix& operator/=( scalar );

        /// @brief  Matrix addition, A + B.
        ///         Equivalent to axpy( x, 1, different ). If you know your
        ///         matrices have identical nonzero patterns, consider
        ///         using axpy instead
        /// \param[in] Matrix   Matrix to add
        Matrix& operator+=( const Matrix& );
        /// @brief      Matrix subtraction, A - B.
        ///             Equivalent to axpy( x, -1, different ). If you know
        ///             your matrices have identical nonzero patterns,
        ///             consider using axpy instead.
        /// \param[in] Matrix   Matrix to add
        Matrix& operator-=( const Matrix& );

        /// @brief  A *= B, where A and B are matrices
        /// Equivalent to multiply( B );
        /// \param[in] B        The Matrix B
        Matrix& operator*=( const Matrix& );

        /* these currently do not have to be friends (since Matrix is
         * implicitly convertible to Mat), but they are for least possible
         * surprise
         */
        friend Matrix operator*( Matrix, scalar );
        friend Matrix operator*( scalar, Matrix );
        friend Matrix operator/( Matrix, scalar );

        friend Matrix operator+( Matrix, const Matrix& );
        friend Matrix operator-( Matrix, const Matrix& );
        /// @brief  C = A * B
        /// Equivalent to multiply( A, B )
        /// \param[in] A    The Matrix A
        /// \param[in] B    The Matrix B
        /// \return    C    The Matrix C
        friend Matrix operator*( const Matrix&, const Matrix& );

        /// @brief  y = Ax, Matrix-Vector multiplication
        /// Equivalent to multiply( A, x )
        friend Vector operator*( const Matrix&, const Vector& );
        friend Vector operator*( const Vector&, const Matrix& );

        /// @brief Tests if two matrices are identical.
        ///
        ///         This also checks Matrix structure, so two identical
        ///         sparse matrices with different nonzero structure but
        ///         with explicit zeros will evaluate to false
        friend bool identical( const Matrix&, const Matrix& );

        /// @brief A += aB
        /// \param[in] B        The Matrix B
        /// \param[in] alpha    The constant alpha
        /// \param[in] pattern  The nonzero pattern relationship
        Matrix& axpy( const Matrix&, scalar, nonzero_pattern );

        /// @brief  A += B.
        /// Equivalent to axpy( Matrix, 1, pattern )
        /// \param[in] Matrix   The Matrix X
        /// \param[in] alpha    The constant alpha
        /// \param[in] pattern  The nonzero pattern relationship
        Matrix& xpy( const Matrix&, nonzero_pattern );

        /// @brief  A *= B.
        ///         Assumes, if sparse, that A and B have the same nonzero pattern
        /// \param[in] B        The Matrix B
        Matrix& multiply( const Matrix& );

        /// @brief  A *= B
        ///         Assumes, if sparse, that A and B have the same nonzero pattern
        /// \param[in] B        The Matrix B
        /// \param[in] fill     The nonzero fill ratio
        /// nnz(C)/( nnz(A) + nnz(B) )
        Matrix& multiply( const Matrix&, scalar fill );

        /// @brief In-place transposes the Matrix, A <- A^T
        Matrix& transpose();

        /// @brief In-place hermitian transposes the Matrix, A <- A^H
        Matrix& hermitian_transpose();

    private:
        Matrix( size_type, size_type );
};

/// @brief  Matrix-Matrix multiplication.
///         This is similar to operator*, but with more flexibility as you
///         can tune the fill. The expected fill ratio of the multiplication
///         C = A * B is nonzero(C)/(nonzero(A)+nonzero(B)).
///
///         Defaults to letting PETSc decide.
///         If experimenting, using PETSc's -info option can print the correct
///         ratio (under fill ratio)
/// \param[in] A    Matrix A (left-hand side)
/// \param[in] B    Matrix B (right-hand side)
/// \param[in] fill (Optional) Matrix fill ratio
Matrix multiply( const Matrix&, const Matrix&, Matrix::scalar = PETSC_DECIDE );

/// @brief y = Ax, Matrix-Vector multiplication.
/// \param[in] A    The Matrix
/// \param[in] x    The Vector
/// \return    y    The result Vector
Vector multiply( const Matrix&, const Vector& );

/// @brief A^T
/// \param[in]  A   Matrix
/// \return     A^T The transposed Matrix A
Matrix transpose( const Matrix& );
/// @brief A^H
/// \param[in]  A   Matrix
/// \return     A^H The hermetian transposed Matrix A
Matrix hermitian_transpose( Matrix );

/// @brief  Non-fully constructed Matrix
///         See \see { Matrix::Builder::Inserter Matrix::Builder::Accumulator }
///         for the implementations of this concept.
///
///         The motivation behind this class is to explicitly separate the
///         Matrix construction and assembly phase from a ready-to-use
///         Matrix. This provides static and tools that prevents possibly
///         ill-formed code from compiling, as well as giving the
///         and user more information regarding intent.
///
///         The general procedure is:
///         #1: determine dimensions of the Matrix.
///         #2: construct a Matrix::Builder with these dimensions. This is
///             unchangeable once the Builder is created.
///         #3: set non-zero coordinates in the Matrix, possibly with a
///             value. See Matrix::Builder::insert.
///         #4: construct the completed Matrix either by returning the
///             Builder, passing it to Matrix' constructor or by calling
///             commit() on the Builder.
///         In code, this would be:
///         \code{.cpp}
///         Matrix create_Matrix( const Grid& grid ) {
///             auto dim = grid.dimensions();
///             Matrix::Builder::Inserter Builder( dim, dim );
///
///             for( auto g : grid ) {
///                 /* perform passes over your data and determine nonzero
///                  * coordinates in the Matrix
///                  */
///                 Builder.insert( g.x, g.y );
///                 Builder.insert( g.y, g.x, g.val() );
///             }
///
///         // move constructor - will reuse the Builders' resources
///         return Builder;
///         }
///         \endcode
///
///         Please note that to achieve performance you have to provide
///         information on the Matrix' nonzero pattern and call the
///         approperiate constructor. In many cases this means an extra
///         pass over your data, but it can result in massive speedup in
///         in Matrix allocation.

class Matrix::Builder : private Matrix {
    /*
     * This is more of a "namespace" class and implementation detail rather
     * than something to be exposed. All constructors are private, so it is
     * obviously not meant to be instantiated by some user. See the Inserter
     * and Accumulator classes which are user-facing.
     */
    public:
        class Accumulator;
        class Inserter;

        typedef Matrix::scalar scalar;
        typedef Matrix::size_type size_type;
    private:
        /// @internal
        /// @copydoc Matrix::Builder::Accumulator( size_type, size_type )
        /// @endinternal
        Builder( size_type, size_type );

        /// @internal
        /// @copydoc Matrix::Builder::Accumulator( size_type,
        ///                                        const std::vector< size_type >& )
        /// @endinternal
        Builder( size_type, size_type, const std::vector< size_type >& );

        /// @internal
        /// @copydoc Matrix::Builder::Accumulator( size_type, size_type,
        ///     const std::vector< size_type >&,
        ///     const std::vector< size_type >& )
        /// @endinternal
        Builder(    size_type, size_type,
                    const std::vector< size_type >&,
                    const std::vector< size_type >& );

        /// @internal
        /// @copydoc Matrix::Builder::Accumulator( const Accumulator& )
        /// @endinternal
        Builder( const Builder& );
        Builder( Builder& );
        Builder( Builder&& );

        Builder( const Accumulator& );
        Builder( Accumulator& );
        Builder( Accumulator&& );

        Builder( const Inserter& );
        Builder( Inserter& );
        Builder( Inserter&& );

        /*
         * I don't actually want to support operator= for PETSc types, but they
         * are defined nevertheless out of necessity. A pattern that emerges in
         * plenty of OPM code already is an init() separate from the
         * constructor. This raises the need of a "default" constructor, but
         * this is out of the question for Builders. A workaround is to
         * construct a zero-sized Matrix and later move-assign into that
         * object.
         *
         * The obvious solution to this is to rewrite the (broken) code, but as
         * operator= is provided as a temporary measure. The use of this
         * operator is highly discouraged.
         */

        Builder& operator=( Builder&& rhs ) {
            static_cast< Matrix& >( *this ) = std::move( static_cast< Matrix& >( rhs ) );
            return *this;
        }

        /// @internal
        /// @brief  Insert a single value.
        /// Defaults to 0.
        /// \param[in] row      The row to insert at
        /// \param[in] col      The col to insert at
        /// \param[in] value    (Optional) the value to insert.
        /// @endinternal
        void at( size_type, size_type, scalar, InsertMode );

        /// @internal
        /// @brief  Insert a full (sub)Matrix in CSR format.
        ///         This is more efficient than inserting single values.
        ///         If the (sub)Matrix sets some value previously set,
        ///         that value will be overwritten by the value in the
        ///         latest call.
        ///
        ///         This might in the future be superseded by a stricter CSR type
        ///
        ///         The row indices Vector is of length rows+1, and with a
        ///         [begin,end) pattern.
        /// \param[in] nonzero     The CSR nonzero Vector
        /// \param[in] row_indices The CSR rowindices Vector
        /// \param[in] col_indices The CSR colindices Vector
        /// @endinternal
        void at(    const std::vector< scalar >& nonzeros,
                    const std::vector< size_type >& row_indices,
                    const std::vector< size_type >& col_indices,
                    InsertMode );

        /// @internal
        /// @brief  Insert a row into the Matrix.
        ///         Inserts from [begin, begin + vec.size())
        ///         This does not overwrite anything beyond vec.size()
        /// \param[in] row      Row to insert
        /// \param[in] values   Values to insert
        /// \param[in] begin    Where in the row to start insertion
        /// @endinternal
        void row(   size_type,
                    const std::vector< scalar >&,
                    size_type,
                    InsertMode );

        /// @internal
        /// @brief  Insert a row into the Matrix at specific indices.
        ///         This does not overwrite anything not touched by the column
        ///         indices Vector. Values must be equal in size to columns.
        /// \param[in] row      Row to insert
        /// \param[in] columns  Vector of column indices to insert
        /// \param[in] values   Values to insert
        /// @endinternal
        void row(   size_type,
                    const std::vector< size_type >&,
                    const std::vector< scalar >&,
                    InsertMode );

        /// @internal
        /// Assembles the Matrix to a finalized state, where no more elements
        /// can be added.
        /// @endinternal
        Builder& assemble();
        /// @internal
        /// Flushes the cache, making it possible to mix add and insert
        /// operations.
        /// @endinternal
        Builder& flush();

        friend class Matrix;
        friend Matrix commit( const Builder& );
        friend Matrix commit( Builder&& );
};

/// @brief  Commit the currently built structure into a completed Matrix.
///         This copies the currently built structure and returns a
///         completed, ready-to-use Matrix. Ideal for when several matrices
///         have identical submatrices.
Matrix commit( const Matrix::Builder& );

/// @brief  Commit the currently built structure into a completed Matrix.
///         This moves the currently built structure and returns a
///         completed, ready-to-use Matrix, and leaves the builder in a valid,
///         but unspecified state. Do not use the builder after move-commit has
///         been performed.
Matrix commit( Matrix::Builder&& );


/*
 * The inheritance here is an implementation detail only. It handles nifty
 * things such as the commit method and removes the need for explicit
 * constructors, which is nice. the Builder class is a shared implementation
 * and namespace only. This applies to the Inserter too.
 */
/// @brief Accumulating Matrix Builder, i.e. A[n,m] += v
///
/// This Builder class sets by values and submatrices by accumulation, and is
/// primarily intended for incremental construction as a staging area. Ideally
/// you should construct your system through Matrix operations, but sometimes
/// that solution is infeasible.
///
/// All operations supported by this Builder perform a += on the cell(s) in
/// question. Obviously, by giving it a negative value it would become a -=.
///
/// Use commit() or pass the object as an argument to Matrix' to finalize the
/// Matrix when it is ready for use.
class Matrix::Builder::Accumulator : public Matrix::Builder {
    private:
        friend class Matrix::Builder::Inserter;
        friend class Matrix::Builder;

    public:
        typedef Matrix::scalar scalar;
        typedef Matrix::size_type size_type;

        /*
         * GCC4.4 does not support inheriting constructors (but it is a C++11
         * feature), so we have to go through the (tedious) job of declaring
         * them ourselves. Their implementation, however, is derived from
         * Matrix::Builder, so it should simply be forwarding calls.
         *
         * The entire multiple inheritance shenanigans will disappear once
         * the full C++11 standard can be used.
         */

        /* We derive copy and move constructors from Builder::Builder by
         * performing a combined forward-and-cast to its constructor
         */
        template< typename T >
        Accumulator( T&& x ) : Builder( std::forward< T >( x ) ) {}

        /// @brief  Constructor.
        ///         PETSc has to know the dimensions of the Matrix
        ///         beforehand, so a default constructor is not provided.
        ///         This constructor performs a guess on nonzeros (for
        ///         preallocation) and is likely to be inefficient. If
        ///         you can provide more information regarding nonzero
        ///         pattern, please use a different constructor.
        /// \param[in] rows Number of rows
        /// \param[in] cols Number of columns
        Accumulator( size_type, size_type );

        /// @brief  Nonzeros per row hinted constructor
        ///         Suggest the number of nonzeros per row, where each
        ///         index in the Vector corresponds to that row, i.e.
        ///         vec[ 0 ] => row 0. Consider the Matrix
        ///             0 0 1 0
        ///             2 3 0 0
        ///             0 0 0 0
        ///             4 0 0 0
        ///         The ideal constructor would then be
        ///         Builder( 4, 4, { 1, 2, 0, 1 } );
        /// \param[in] rows     Number of rows
        /// \param[in] cols     Number of columns
        /// \param[in] nonzeros Nonzeros per row.
        Accumulator( size_type, size_type, const std::vector< size_type >& );

        /// @brief  Nonzeros on/off diagonal hinted constructor
        ///         Suggest the (average) number of nonzeros on and off the
        ///         diagonal portion of the Matrix. Consider the Matrix
        ///
        ///             A | B | C
        ///             --+---+---
        ///             D | E | F
        ///             --+---+---
        ///             G | H | I
        ///
        ///         Where the letters are sub matrices. The diagonal
        ///         portion are the matrices A, E and I. Please refer to
        ///         the PETSc documentation for a slightly more elaborate
        ///         example: http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMPIAIJSetPreallocation.html
        ///
        ///         This function is currently not MPI aware, so if you
        ///         want MPI aware allocation, determining which process
        ///         owns which subMatrix and only give it those coordinates,
        ///         must be handled by the user.
        ///
        /// \param[in] rows     Number of rows.
        /// \param[in] cols     Number of columns.
        /// \param[in] diag     Vector of nonzeros per row on the diagonal
        ///                     portion.
        /// \param[in] offdiag  Vector of nonzeros per row on the
        ///                      off-diagonal portion of the Matrix.
        Accumulator(    size_type, size_type,
                        const std::vector< size_type >&,
                        const std::vector< size_type >& );

        /// @brief  Insert a single value.
        /// Defaults to 0.
        /// \param[in] row      The row to insert at
        /// \param[in] col      The col to insert at
        /// \param[in] value    (Optional) the value to insert.
        Accumulator& add( size_type, size_type, scalar = scalar() );

        /// @brief  Insert a full (sub)Matrix in CSR format.
        ///         This is more efficient than inserting single values.
        ///         If the (sub)Matrix sets some value previously set,
        ///         that value will be overwritten by the value in the
        ///         latest call.
        ///
        ///         This might in the future be superseded by a stricter CSR type
        ///
        ///         The row indices Vector is of length rows+1, and with a
        ///         [begin,end) pattern.
        /// \param[in] nonzero     The CSR nonzero Vector
        /// \param[in] row_indices The CSR rowindices Vector
        /// \param[in] col_indices The CSR colindices Vector
        Accumulator& add(  const std::vector< scalar >& nonzeros,
                const std::vector< size_type >& row_indices,
                const std::vector< size_type >& col_indices );

        /// @brief  Insert a row into the Matrix.
        ///         Inserts from [begin, begin + vec.size())
        ///         This does not overwrite anything beyond vec.size()
        /// \param[in] row      Row to insert
        /// \param[in] values   Values to insert
        /// \param[in] begin    Where in the row to start insertion
        Accumulator& add_row( size_type,
                const std::vector< scalar >&,
                size_type = 0 );

        /// @brief  Insert a row into the Matrix at specific indices.
        ///         This does not overwrite anything not touched by the column
        ///         indices Vector. Values must be equal in size to columns.
        /// \param[in] row      Row to insert
        /// \param[in] columns  Vector of column indices to insert
        /// \param[in] values   Values to insert
        Accumulator& add_row( size_type,
                    const std::vector< size_type >&,
                    const std::vector< scalar >& );
};
/// @brief Inserting Matrix Builder, i.e. A[n,m] = v
///
/// This Builder class sets by values and submatrices by insertion. It is the
/// preferred method of building matrices, however, Matrix::Builder::Accumulator
/// is provided if your problem requires an accumulating staging area.
///
/// All operations supported by this Builder perform an assignment on the
/// cell(s) in question. Repeated insertions in the same cell turns into the
/// value set last.
///
/// Use commit() or pass the object as an argument to Matrix' to finalize the
/// Matrix when it is ready for use.

class Matrix::Builder::Inserter : public Matrix::Builder {
    private:
        friend class Matrix::Builder::Accumulator;
        friend class Matrix::Builder;

    public:
        typedef Matrix::scalar scalar;
        typedef Matrix::size_type size_type;

        template< typename T >
        Inserter( T&& x ) : Builder( std::forward< T >( x ) ) {}

        /*
         * GCC4.4 does not support inheriting constructors (but it is a C++11
         * feature), so we have to go through the (tedious) job of declaring
         * them ourselves. Their implementation, however, is derived from
         * Matrix::Builder, so it should simply be forwarding calls.
         *
         * The entire multiple inheritance shenanigans will disappear once
         * the full C++11 standard can be used.
         */

        /// @brief  Constructor.
        ///         PETSc has to know the dimensions of the Matrix
        ///         beforehand, so a default constructor is not provided.
        ///         This constructor performs a guess on nonzeros (for
        ///         preallocation) and is likely to be inefficient. If
        ///         you can provide more information regarding nonzero
        ///         pattern, please use a different constructor.
        /// \param[in] rows Number of rows
        /// \param[in] cols Number of columns
        Inserter( size_type, size_type );

        /// @brief  Nonzeros per row hinted constructor
        ///         Suggest the number of nonzeros per row, where each
        ///         index in the Vector corresponds to that row, i.e.
        ///         vec[ 0 ] => row 0. Consider the Matrix
        ///             0 0 1 0
        ///             2 3 0 0
        ///             0 0 0 0
        ///             4 0 0 0
        ///         The ideal constructor would then be
        ///         Builder( 4, 4, { 1, 2, 0, 1 } );
        /// \param[in] rows     Number of rows
        /// \param[in] cols     Number of columns
        /// \param[in] nonzeros Nonzeros per row.
        Inserter( size_type, size_type, const std::vector< size_type >& );

        /// @brief  Nonzeros on/off diagonal hinted constructor
        ///         Suggest the (average) number of nonzeros on and off the
        ///         diagonal portion of the Matrix. Consider the Matrix
        ///
        ///             A | B | C
        ///             --+---+---
        ///             D | E | F
        ///             --+---+---
        ///             G | H | I
        ///
        ///         Where the letters are sub matrices. The diagonal
        ///         portion are the matrices A, E and I. Please refer to
        ///         the PETSc documentation for a slightly more elaborate
        ///         example: http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMPIAIJSetPreallocation.html
        ///
        ///         This function is currently not MPI aware, so if you
        ///         want MPI aware allocation, determining which process
        ///         owns which subMatrix and only give it those coordinates,
        ///         must be handled by the user.
        ///
        /// \param[in] rows     Number of rows.
        /// \param[in] cols     Number of columns.
        /// \param[in] diag     Vector of nonzeros per row on the diagonal
        ///                     portion.
        /// \param[in] offdiag  Vector of nonzeros per row on the
        ///                      off-diagonal portion of the Matrix.
        Inserter(       size_type, size_type,
                        const std::vector< size_type >&,
                        const std::vector< size_type >& );

        /// @brief  Insert a single value.
        /// Defaults to 0.
        /// \param[in] row      The row to insert at
        /// \param[in] col      The col to insert at
        /// \param[in] value    (Optional) the value to insert.
        Inserter& insert( size_type, size_type, scalar = scalar() );

        /// @brief  Insert a full (sub)Matrix in CSR format.
        ///         This is more efficient than inserting single values.
        ///         If the (sub)Matrix sets some value previously set,
        ///         that value will be overwritten by the value in the
        ///         latest call.
        ///
        ///         This might in the future be superseded by a stricter CSR type
        ///
        ///         The row indices Vector is of length rows+1, and with a
        ///         [begin,end) pattern.
        /// \param[in] nonzero     The CSR nonzero Vector
        /// \param[in] row_indices The CSR rowindices Vector
        /// \param[in] col_indices The CSR colindices Vector
        Inserter& insert(   const std::vector< scalar >& nonzeros,
                            const std::vector< size_type >& row_indices,
                            const std::vector< size_type >& col_indices );

        /// @brief  Insert a row into the Matrix.
        ///         Inserts from [begin, begin + vec.size())
        ///         This does not overwrite anything beyond vec.size()
        /// \param[in] row      Row to insert
        /// \param[in] values   Values to insert
        /// \param[in] begin    Where in the row to start insertion
        Inserter& insert_row(   size_type,
                                const std::vector< scalar >&,
                                size_type = 0 );

        /// @brief  Insert a row into the Matrix at specific indices.
        ///         This does not overwrite anything not touched by the column
        ///         indices Vector. Values must be equal in size to columns.
        /// \param[in] row      Row to insert
        /// \param[in] columns  Vector of column indices to insert
        /// \param[in] values   Values to insert
        Inserter& insert_row(   size_type,
                                const std::vector< size_type >&,
                                const std::vector< scalar >& );
};

}
}

#endif //OPM_PETSCMATRIX_H
