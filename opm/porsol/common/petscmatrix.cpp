#include <opm/core/linalg/petsc.hpp>
#include <opm/core/linalg/petscvector.hpp>
#include <opm/core/linalg/petscmatrix.hpp>


#include <algorithm>
#include <cassert>
#include <memory>
#include <utility>
#include <vector>

template< typename T = Opm::Petsc::Matrix::size_type, typename U >
static inline std::vector< T > range( U begin, U end ) {

    std::vector< T > x( end - begin );

    T value( 0 );
    auto first = x.begin();
    const auto last = x.end();

    while( first != last ) {
        *first++ = value++;
    }

    return x;
}

namespace Opm {
namespace Petsc {

/*
 * Trivial translation between the strongly typed enum and the
 * implicit-convertible MatStructure enum.
 */
static inline MatStructure
nz_structure( Matrix::nonzero_pattern x ) {
    switch( x ) {
        case Matrix::nonzero_pattern::different:
            return DIFFERENT_NONZERO_PATTERN;

        case Matrix::nonzero_pattern::subset:
            return SUBSET_NONZERO_PATTERN;

        case Matrix::nonzero_pattern::same:
            return SAME_NONZERO_PATTERN;
    }

    assert( false );
    /* To silence non-void without return statement warning. This should never
     * be reached anyways, and is a sure error.
     */
    throw;
}

static inline Mat default_matrix() {
    /* Meant to be used by other constructors. */
    Mat x;
    MatCreate( Petsc::comm(), &x );
    MatSetFromOptions( x );

    return x;
}

static inline Mat copy_Matrix( Mat x ) {
    Mat y;
    auto err = MatConvert( x, MATSAME, MAT_INITIAL_MATRIX, &y );
    std::unique_ptr< _p_Mat, deleter< _p_Mat > > result( y ); CHKERRXX( err );

    return result.release();
}

static inline
Mat sized_matrix( Matrix::size_type rows, Matrix::size_type cols ) {
    std::unique_ptr< _p_Mat, deleter< _p_Mat > > result( default_matrix() );
    auto err = MatSetSizes( result.get(),
            PETSC_DECIDE, PETSC_DECIDE,
            rows, cols );
    CHKERRXX( err );

    return result.release();
}

Matrix::Matrix( Mat x ) : uptr< Mat >( x ) {}

Matrix::Matrix( const Matrix& x ) : uptr< Mat >( copy_Matrix( x ) ) {}

/*
 * These constructors are unfortunately slightly complex as full C++11 support
 * isn't available.
 *
 * In C++11 lingo this is basically using a delegating constructor to Matrix(
 * Matrix&& ) where Matrix has using uptr::uptr set, i.e. the inherited
 * constructor. This is complicated further by the
 * implicit-conversion-to-pointer support, which means that just passing
 * Matrix& to a constructor could trigger the implicit cast, which happens to
 * be a valid uptr constructor.
 *
 * However, the correct behaviour can be ensured by casting. We know that no
 * classes actually add any fields to the parent class, and we know (by
 * convention) that it's the parent class that handles ownership. This means
 * that simply upcasting the pointer to pick the right constructor overload is
 * sufficient, relying on that constructor to handle ownership. Some complexity
 * is (necessarily so) added by move semantics, but the benefit here is
 * obvious.
 */
Matrix::Matrix( Matrix&& x ) :
    uptr< Mat >( std::move( static_cast< uptr< Mat >& >( x ) ) )
{}

Matrix& Matrix::operator=( const Matrix& rhs ) {
    Matrix rhs_copy( rhs );
    this->swap( rhs_copy );
    return *this;
}

Matrix& Matrix::operator=( Matrix&& rhs ) {
    this->swap( rhs );
    return *this;
}

Matrix::Matrix( const std::vector< Matrix::scalar >& values,
                Matrix::size_type rows,
                Matrix::size_type cols ) :
        uptr< Mat >( sized_matrix( rows, cols ) )
{
    /* this is a very specific load to create a dense Matrix */
    MatSetType( this->ptr(), MATSEQDENSE );

    auto err = MatSeqDenseSetPreallocation( this->ptr(), NULL );
    CHKERRXX( err );

    const auto indices = range( 0, std::max( rows, cols ) );

    err = MatSetValues( this->ptr(),
                    rows, indices.data(),
                    cols, indices.data(),
                    values.data(), INSERT_VALUES );
    CHKERRXX( err );

    err = MatAssemblyBegin( this->ptr(), MAT_FINAL_ASSEMBLY );
    CHKERRXX( err );
    err = MatAssemblyEnd( this->ptr(), MAT_FINAL_ASSEMBLY );
    CHKERRXX( err );
}

Matrix::size_type Matrix::rows() const {
    PetscInt x;
    auto err = MatGetSize( this->ptr(), &x, NULL ); CHKERRXX( err );
    return x;
}

Matrix::size_type Matrix::cols() const {
    PetscInt x;
    auto err = MatGetSize( this->ptr(), NULL, &x ); CHKERRXX( err );
    return x;
}

Matrix& Matrix::operator*=( Matrix::scalar rhs ) {
    auto err = MatScale( this->ptr(), rhs ); CHKERRXX( err );
    return *this;
}

Matrix& Matrix::operator/=( Matrix::scalar rhs ) {
    return *this *= ( 1 / rhs );
}

Matrix& Matrix::operator+=( const Matrix& rhs ) {
    return this->axpy( rhs, 1, nonzero_pattern::different );
}

Matrix& Matrix::operator-=( const Matrix& rhs ) {
    return this->axpy( rhs, -1, nonzero_pattern::different );
}

Matrix& Matrix::operator*=( const Matrix& rhs ) {
    return this->multiply( rhs );
}

Matrix operator*( Matrix lhs, Matrix::scalar rhs ) {
    return lhs *= rhs;
}

Matrix operator*( Matrix::scalar lhs, Matrix rhs ) {
    return rhs *= lhs;
}

Matrix operator/( Matrix lhs, Matrix::scalar rhs ) {
    return lhs /= rhs;
}

Matrix operator+( Matrix lhs, const Matrix& rhs ) {
    return lhs += rhs;
}

Matrix operator-( Matrix lhs, const Matrix& rhs ) {
    return lhs -= rhs;
}

Matrix operator*( const Matrix& lhs, const Matrix& rhs ) {
    return multiply( lhs, rhs );
}

Vector operator*( const Matrix& lhs, const Vector& rhs ) {
    return multiply( lhs, rhs );
}

Vector operator*( const Vector& lhs, const Matrix& rhs ) {
    return multiply( rhs, lhs );
}

bool identical( const Matrix& lhs, const Matrix& rhs ) {
    /* Comparing a Matrix to itself means they're obviously identical */
    if( lhs.ptr() == rhs.ptr() ) return true;

    /* PETSc throws an exception if matrices are of different sizes. Because
     * two matrices of different sizes -cannot- be equal, we check this first
     */
    PetscInt row_lhs, col_lhs, row_rhs, col_rhs;
    auto err = MatGetSize( lhs, &row_lhs, &col_lhs ); CHKERRXX( err );
    err = MatGetSize( rhs, &row_rhs, &col_rhs ); CHKERRXX( err );

    if( row_lhs != row_rhs ) return false;
    if( col_lhs != col_rhs ) return false;

    /* MatEqual also considers structure when testing for equality. See:
     * http://lists.mcs.anl.gov/pipermail/petsc-users/2015-January/024059.html
     */
    PetscBool eq;
    err = MatEqual( lhs, rhs, &eq ); CHKERRXX( err );
    return eq;
}

Matrix& Matrix::axpy(
        const Matrix& x,
        Matrix::scalar a,
        Matrix::nonzero_pattern nz ) {

    auto err = MatAXPY( this->ptr(), a, x, nz_structure( nz ) );
    CHKERRXX( err );

    return *this;
}

Matrix& Matrix::xpy( const Matrix& x, Matrix::nonzero_pattern nz ) {
    return this->axpy( x, 1, nz );
}

Matrix& Matrix::multiply( const Matrix& x ) {
    auto err = MatMatMult( this->ptr(), x,
            MAT_REUSE_MATRIX, PETSC_DEFAULT, NULL );
    CHKERRXX( err );

    return *this;
}

Matrix& Matrix::transpose() {
    Mat handle = this->ptr();
    auto err = MatTranspose( handle, MAT_REUSE_MATRIX, &handle );
    CHKERRXX( err );

    return *this;
}

Matrix& Matrix::hermitian_transpose() {
    auto handle = this->ptr();
    auto err = MatHermitianTranspose( handle,
            MAT_REUSE_MATRIX, &handle );
    CHKERRXX( err );

    return *this;
}

Matrix multiply( const Matrix& rhs, const Matrix& lhs, Matrix::scalar fill ) {
    Mat x;
    MatMatMult( rhs, lhs, MAT_INITIAL_MATRIX, fill, &x );

    return Matrix( x );
}

Vector multiply( const Matrix& lhs, const Vector& rhs ) {
    Vec x;
    auto err = VecDuplicate( rhs, &x ); CHKERRXX( err );
    MatMult( lhs, rhs, x );
    return Vector( x );
}

Matrix transpose( const Matrix& rhs ) {
    Mat x;
    MatTranspose( rhs, MAT_INITIAL_MATRIX, &x );
    return Matrix( x );
}

Matrix hermitian_transpose( Matrix rhs ) {
    return rhs.hermitian_transpose();
}

Matrix::Builder::Builder( Matrix::size_type rows, Matrix::size_type cols )
    : Matrix( sized_matrix( rows, cols ) )
{

    /*
     * This is an awkward case - we haven't been given much preallocation
     * information to use, so we guess. Using the Builder without setting
     * proper preallocation may be considered an error in the future.
     */

    /* first, to avoid overflow, rescale the Matrix dimensions to 1% of its
     * size. If the Matrix is smaller than 10x10 (unlikely, except for test
     * cases), just set the new dimensions to 1.
     */
    const int m = std::max( 1, rows / 10 );
    const int n = std::max( 1, cols / 10 );

    /*
     * Guess nonzeros per row in the diagonal portion of the Matrix. Consider
     * the Matrix
     * A | B | C
     * --+---+---
     * D | E | F
     * --+---+---
     * G | H | I
     *
     * Where the letters are sub matrices. The diagonal portion are the
     * matrices A, E and I.
     */

    /*
     * Per row in the diagonal portion we're again assuming 1% fill -of the 1%
     * Matrix-, i.e. for a 100k row Matrix, we're assuming 10 nonzeros per row.
     * Falling back to a minimum of 1. We're assuming we have the same number
     * off the diagonal as well.
     */
    const int nnz_on_diag = std::max( 1, ( m * n ) / 100 );

    auto err = MatSeqAIJSetPreallocation( this->ptr(),
            2 * nnz_on_diag, NULL );
    CHKERRXX( err );

    err = MatMPIAIJSetPreallocation( this->ptr(),
            nnz_on_diag, NULL,
            nnz_on_diag, NULL );

    CHKERRXX( err );
}

Matrix::Builder::Builder(
        Matrix::size_type rows,
        Matrix::size_type cols,
        const std::vector< Matrix::size_type >& nnz_per_row )
    : Matrix( sized_matrix( rows, cols ) )
{

    assert( nnz_per_row.size() == rows );

    auto err = MatSeqAIJSetPreallocation( this->ptr(),
            /*ignored*/ 0,
            nnz_per_row.data() );
    CHKERRXX( err );

    /* Some shenanigans to minimise communication during Matrix assembly.
     *
     * It is now sufficient to extract information for the parts that will end
     * up belonging to this processor. In the worst case - the sequential one -
     * it ends up being one full array copy.
     */

    Matrix::size_type begin, end;
    MatGetOwnershipRange( this->ptr(), &begin, &end );

    std::vector< Matrix::size_type > nnz_diag(
            nnz_per_row.begin() + begin,
            nnz_per_row.begin() + end );

    /*
     * This scheme overcommits local memory by a factor of 2 - that is, if the
     * nnz_per_row Vector is precise, preallocation will allocate exactly twice
     * the memory, because it assumes as many nonzeros per row in the
     * off-diagonals as in the diagonals. If this assumption does not hold and
     * you want less memory strain, use a different constructor and give it
     * more information.
     */
    err = MatMPIAIJSetPreallocation( this->ptr(),
            /* ignored */ 0, nnz_diag.data(),
            /* ignored */ 0, nnz_diag.data() );

    CHKERRXX( err );
}

Matrix::Builder::Builder(
        Matrix::size_type rows,
        Matrix::size_type cols,
        const std::vector< Matrix::size_type >& nnz_on_diag,
        const std::vector< Matrix::size_type >& nnz_off_diag )
    : Matrix( sized_matrix( rows, cols ) )
{

    assert( nnz_on_diag.size() == nnz_off_diag.size() );

    /* nnz_per_row[ i ] = nnz_on_diag[ i ] + nnz_off_diag[ i ]; */
    std::vector< Matrix::size_type > nnz_per_row( nnz_on_diag.size() );
    std::transform( nnz_on_diag.begin(), nnz_on_diag.end(),
            nnz_off_diag.begin(), nnz_per_row.begin(),
            std::plus< Matrix::size_type >() );

    /*
     * We calculate the sequential version's nonzero-per-row by simply
     * zip-summing the on- and off-diag Vectors
     */
    auto err = MatSeqAIJSetPreallocation( this->ptr(),
            /*ignored*/ 0,
            nnz_per_row.data() );
    CHKERRXX( err );

    /* Set only the local values to reduce communication */
    Matrix::size_type offset;
    MatGetOwnershipRange( this->ptr(), &offset, NULL );

    err = MatMPIAIJSetPreallocation( this->ptr(),
            /* ignored */ 0, nnz_on_diag.data() + offset,
            /* ignored */ 0, nnz_off_diag.data() + offset );

    CHKERRXX( err );
}

static inline Mat copy_Builder( Mat x ) {
    /*
     * We know this is supposed to end up in another Builder, but we must still
     * use MAT_FINAL_ASSEMBLY, because the operations MatDuplicate and MatCopy
     * won't work unless the Matrix is fully assembled. As in the similar
     * Matrix functions, this uses std::unique_ptr to provide exception safety.
     */
    auto err = MatAssemblyBegin( x, MAT_FINAL_ASSEMBLY );
    CHKERRXX( err );
    err = MatAssemblyEnd( x, MAT_FINAL_ASSEMBLY );
    CHKERRXX( err );

    Mat y;
    err = MatDuplicate( x, MAT_COPY_VALUES, &y );
    std::unique_ptr< _p_Mat, deleter< _p_Mat > > result( y );
    CHKERRXX( err );
    err = MatCopy( x, y, DIFFERENT_NONZERO_PATTERN ); CHKERRXX( err );

    return result.release();
}

/*
 * More casting fun! See the comments of Matrix::Matrix for reasoning.
 *
 * These constructors rely on Matrix to deal with ownership etc., so we really
 * only concern ourselves with ensuring well-definedness and a consistent,
 * correct state. We know that these constructors will result in other
 * Builders, so in the case of moves we do not have to actually assemble the
 * Matrix properly - rather, flushing it is sufficient to handle the case where
 * Accumulators are mixed with Inserters.
 *
 * Builder copies uses the helper function copy_Builder, as a Matrix must be
 * fully assembled before duplicate+copy can be used. However, at that point we
 * know that the copy has been performed and that we have a new Matrix handle,
 * so we just use the take-ownership-of-raw-pointer constructor.
 *
 * Since all these constructors rely on Matrix::Matrix for ownership, it is
 * sufficient to call that constructor and cast to Matrix& + move
 *
 * On top of all of this we're hit by the (ugly) case of too-perfect
 * forwarding, which would again be resolved slightly by delegating
 * constructors. In most cases, the Builders are non-const references passed to
 * eachother, and std::forward would pick the wrong overload in these cases,
 * giving it to the move constructor. We fix (workaround, rather) this by also
 * offering a non-const copy constructor. It is a brute-force approach, and
 * works because it "only" gives three more overloads.
 */
Matrix::Builder::Builder( const Matrix::Builder& x ) :
    Matrix( copy_Builder( x.ptr() ) )
{}

Matrix::Builder::Builder( Matrix::Builder& x ) :
    Matrix( copy_Builder( x.ptr() ) )
{}

Matrix::Builder::Builder( Matrix::Builder&& x ) :
    Matrix( std::move( static_cast< Matrix& >( x.flush() ) ) )
{}

Matrix::Builder::Builder( const Matrix::Builder::Accumulator& x ) :
    Matrix( copy_Builder( x.ptr() ) )
{}
Matrix::Builder::Builder( Matrix::Builder::Accumulator& x ) :
    Matrix( copy_Builder( x.ptr() ) )
{}
Matrix::Builder::Builder( Matrix::Builder::Accumulator&& x ) :
    Matrix( std::move( static_cast< Matrix& >( x.flush() ) ) )
{}

Matrix::Builder::Builder( const Matrix::Builder::Inserter& x ) :
    Matrix( copy_Builder( x.ptr() ) )
{}
Matrix::Builder::Builder( Matrix::Builder::Inserter& x ) :
    Matrix( copy_Builder( x.ptr() ) )
{}
Matrix::Builder::Builder( Matrix::Builder::Inserter&& x ) :
    Matrix( std::move( static_cast< Matrix& >( x.flush() ) ) )
{}

void Matrix::Builder::at(
        Matrix::size_type row,
        Matrix::size_type col,
        Matrix::scalar value,
        InsertMode mode ) {

    const size_type rows[] = { row };
    const size_type cols[] = { col };
    const scalar vals[] = { value };

    MatSetOption(this->ptr(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE); 

	auto err = MatSetValues( this->ptr(),
            1, rows,
            1, cols,
            vals, mode ); CHKERRXX( err );
}

void Matrix::Builder::at(
        const std::vector< Matrix::scalar >& nonzeros,
        const std::vector< Matrix::size_type >& row_indices,
        const std::vector< Matrix::size_type >& col_indices,
        InsertMode mode ) {

    assert( nonzeros.size() == col_indices.size() );

    for( unsigned int i = 0; i < row_indices.size() - 1; ++i ) {
        /* the difference between two consecutive elements are the indices in
         * the col&nonzero Vectors where the current row i is stored
         */
        const auto row_entries = row_indices[ i + 1 ] - row_indices[ i ];

        /* empty row - skip ahead */
        if( !row_entries ) continue;

        /* MatSetValues takes an array */
        const Matrix::size_type row_index[] = { Matrix::size_type( i ) };

        /* offset to start insertion from */
        const auto offset = row_indices[ i ];

        const auto err = MatSetValues( this->ptr(),
                1, row_index,
                row_entries, col_indices.data() + offset,
                nonzeros.data() + offset, mode );

        CHKERRXX( err );
    }
}

void Matrix::Builder::row(
        Matrix::size_type row,
        const std::vector< Matrix::scalar >& values,
        Matrix::size_type begin,
        InsertMode mode ) {

    const auto indices = range( begin, size_type( begin + values.size() ) );

    this->row( row, indices, values, mode );
}

void Matrix::Builder::row(
        Matrix::size_type row,
        const std::vector< Matrix::size_type >& indices,
        const std::vector< Matrix::scalar >& values,
        InsertMode mode ) {

    /* MatSetValues takes an array */
    const Matrix::size_type row_index[] = { row };

    const auto err = MatSetValues( this->ptr(),
            1, row_index,
            indices.size(), indices.data(),
            values.data(), mode );

    CHKERRXX( err );
}

Matrix commit( const Matrix::Builder& x ) {
    /*
     * This is a peculiar case. So commit (and copy) doesn't actually -change-
     * the Builder, but it does require mutability. The flush method calls
     * MatAssemble from PETSc which flushes the caches and sets up the Matrix
     * structure. However, this is an implementation detail, and the object
     * that the Builder exposes is oblivious to this. We use the const_cast
     * trick to -appear- immutable, while we need mutability to flush our
     * caches.
     *
     * This is ok because the -visible- object does not change, and is for all
     * intents and purposes still const
     */
    const_cast< Matrix::Builder& >( x ).assemble();

    /* Ensure we call Matrix( const Matrix& ). If we call Matrix( const
     * Builder& ) we would end up in a loop, because the Matrix( Builder& )
     * copy constructor must also ensure that commit is called.
     */
    return Matrix( static_cast< const Matrix& >( x ) );
}

Matrix commit( Matrix::Builder&& x ) {
    x.assemble();
    return static_cast< Matrix&& >( x );
}

Matrix::Builder& Matrix::Builder::assemble() {
    auto err = MatAssemblyBegin( this->ptr(), MAT_FINAL_ASSEMBLY );
    CHKERRXX( err );
    err = MatAssemblyEnd( this->ptr(), MAT_FINAL_ASSEMBLY );
    CHKERRXX( err );

    return *this;
}

Matrix::Builder& Matrix::Builder::flush() {
    auto err = MatAssemblyBegin( this->ptr(), MAT_FLUSH_ASSEMBLY );
    CHKERRXX( err );
    err = MatAssemblyEnd( this->ptr(), MAT_FLUSH_ASSEMBLY );
    CHKERRXX( err );

    return *this;
}

Matrix::Builder::Accumulator::Accumulator(
        Matrix::size_type rows,
        Matrix::size_type cols ) :
    Matrix::Builder( rows, cols )
{}

Matrix::Builder::Accumulator::Accumulator(
        Matrix::size_type rows,
        Matrix::size_type cols,
        const std::vector< Matrix::size_type >& nonzeros ) :
    Matrix::Builder( rows, cols, nonzeros )
{}

Matrix::Builder::Accumulator::Accumulator(
        Matrix::size_type rows,
        Matrix::size_type cols,
        const std::vector< Matrix::size_type >& ondiag,
        const std::vector< Matrix::size_type >& offdiag ) :
    Matrix::Builder( rows, cols, ondiag, offdiag )
{}


Matrix::Builder::Accumulator& Matrix::Builder::Accumulator::add(
        Matrix::size_type x,
        Matrix::size_type y,
        Matrix::scalar val ) {

    this->at( x, y, val, ADD_VALUES );
    return *this;
}

Matrix::Builder::Accumulator& Matrix::Builder::Accumulator::add(
        const std::vector< scalar >& nonzeros,
        const std::vector< size_type >& row_indices,
        const std::vector< size_type >& col_indices ) {

    this->at( nonzeros, row_indices, col_indices, ADD_VALUES );
    return *this;
}

Matrix::Builder::Accumulator& Matrix::Builder::Accumulator::add_row(
        size_type row,
        const std::vector< scalar >& values,
        size_type begin ) {
    this->row( row, values, begin, ADD_VALUES );
    return *this;
}

Matrix::Builder::Accumulator& Matrix::Builder::Accumulator::add_row(
        size_type row,
        const std::vector< size_type >& cols,
        const std::vector< scalar >& vals ) {
    this->row( row, cols, vals, ADD_VALUES );
    return *this;
}

Matrix::Builder::Inserter::Inserter(
        Matrix::size_type rows,
        Matrix::size_type cols ) :
    Matrix::Builder( rows, cols )
{}

Matrix::Builder::Inserter::Inserter(
        Matrix::size_type rows,
        Matrix::size_type cols,
        const std::vector< Matrix::size_type >& nonzeros ) :
    Matrix::Builder( rows, cols, nonzeros )
{}

Matrix::Builder::Inserter::Inserter(
        Matrix::size_type rows,
        Matrix::size_type cols,
        const std::vector< Matrix::size_type >& ondiag,
        const std::vector< Matrix::size_type >& offdiag ) :
    Matrix::Builder( rows, cols, ondiag, offdiag )
{}


Matrix::Builder::Inserter& Matrix::Builder::Inserter::insert(
        Matrix::size_type x,
        Matrix::size_type y,
        Matrix::scalar val ) {

    this->at( x, y, val, INSERT_VALUES );
    return *this;
}

Matrix::Builder::Inserter& Matrix::Builder::Inserter::insert(
        const std::vector< scalar >& nonzeros,
        const std::vector< size_type >& row_indices,
        const std::vector< size_type >& col_indices ) {

    this->at( nonzeros, row_indices, col_indices, INSERT_VALUES );
    return *this;
}

Matrix::Builder::Inserter& Matrix::Builder::Inserter::insert_row(
        size_type row,
        const std::vector< scalar >& values,
        size_type begin ) {
    this->row( row, values, begin, INSERT_VALUES );
    return *this;
}

Matrix::Builder::Inserter& Matrix::Builder::Inserter::insert_row(
        size_type row,
        const std::vector< size_type >& cols,
        const std::vector< scalar >& vals ) {
    this->row( row, cols, vals, INSERT_VALUES );
    return *this;
}

}
}
