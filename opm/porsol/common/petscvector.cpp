#include <opm/core/linalg/petsc.hpp>
#include <opm/core/linalg/petscvector.hpp>

#include <cassert>
#include <memory>
#include <utility>

#include <iostream>

namespace Opm {
namespace Petsc {

static inline std::vector< Vector::size_type >
range( Vector::size_type begin, Vector::size_type end ) {
    using namespace std;

    std::vector< Vector::size_type > x( end - begin );
    /* with full C++11 support this can be replaced with a call to std::iota */

    Vector::size_type value( 0 );
    auto first = x.begin();
    const auto last = x.end();

    while( first != last ) {
        *first++ = value++;
    }

    return x;
}

static inline Vec default_Vector() {
    /* Meant to be used by other constructors. */
    Vec x;
    VecCreate( Petsc::comm(), &x );
    VecSetFromOptions( x );

    return x;
}

/* To provide exception safety, wrap the Vector handles in a manged
 * (unique_ptr) object. While it looks a bit ugly, it is used basically only in
 * a few functions, so it should be ok.
 *
 * Using delegating constructors (gcc4.7-4.8) would be better, so a future TODO
 * is to remove these functions in favour of delegating constructors.
 */

static inline Vec copy_Vector( Vec x ) {
    Vec y;
    auto err = VecDuplicate( x, &y ); CHKERRXX( err );
    std::unique_ptr< _p_Vec, deleter< _p_Vec > > result( y );
    err = VecCopy( x, y ); CHKERRXX( err );

    return result.release();
}

static inline Vec sized_Vector( Vector::size_type size ) {
    std::unique_ptr< _p_Vec, deleter< _p_Vec > > result( default_Vector() );

    auto err = VecSetSizes( result.get(), PETSC_DECIDE, size );
    CHKERRXX( err );

    return result.release();
}

static inline Vec set_Vector( const std::vector< Vector::scalar >& values,
                            const std::vector< Vector::size_type >& indices ) {

    assert( values.size() == indices.size() );

    std::unique_ptr< _p_Vec, deleter< _p_Vec > >
        result( sized_Vector( values.size() ) );

    auto err = VecSetValues( result.get(), values.size(),
            indices.data(), values.data(),
            INSERT_VALUES );
    CHKERRXX( err );

    err = VecAssemblyBegin( result.get() ); CHKERRXX( err );
    err = VecAssemblyEnd( result.get() ); CHKERRXX( err );

    return result.release();

}

Vector::Vector( Vec x ) : uptr< Vec >( x ) {}

Vector::Vector( const Vector& x ) : uptr< Vec >( copy_Vector( x ) ) {}

Vector::Vector( Vector&& x ) : uptr< Vec >( std::move( x ) ) {}

/*
 * When we can move to full C++11 support, these should ideally be implemented
 * with delegating constructors.
 */

Vector::Vector( Vector::size_type size ) : uptr< Vec >( sized_Vector( size ) ) {}

Vector::Vector( Vector::size_type size, Vector::scalar x ) :
        uptr< Vec >( sized_Vector( size ) )
{
    this->assign( x );
}

Vector::Vector( const std::vector< Vector::scalar >& values ) :
    uptr< Vec >( set_Vector( values, range( 0, values.size() ) ) )
{}

Vector::Vector( const std::vector< Vector::scalar >& values,
                const std::vector< Vector::size_type >& indexset ) :
    uptr< Vec >( set_Vector( values, indexset ) )
{}

Vector::size_type Vector::size() const {
    PetscInt x;
    auto err = VecGetSize( this->ptr(), &x ); CHKERRXX( err );
    return x;
}

void Vector::assign( Vector::scalar x ) {
    auto err = VecSet( this->ptr(), x ); CHKERRXX( err );
}

Vector& Vector::operator+=( Vector::scalar rhs ) {
    auto err = VecShift( this->ptr(), rhs ); CHKERRXX( err );
    return *this;
}

Vector& Vector::operator-=( Vector::scalar rhs ) {
    return *this += -rhs;
}

Vector& Vector::operator*=( Vector::scalar rhs ) {
    auto err = VecScale( this->ptr(), rhs ); CHKERRXX( err );
    return *this;
}

Vector& Vector::operator/=( Vector::scalar rhs ) {
    return *this *= ( 1 / rhs );
}

Vector& Vector::operator+=( const Vector& rhs ) {
    if( this->ptr() == rhs.ptr() ) {
        /* adding something to itself is equivalent to *= 2.
         * VecAXPY breaks if both arguments are the same, so this must be
         * special-cased.
         */
        return *this *= 2;
    }

    auto err = VecAXPY( this->ptr(), 1, rhs ); CHKERRXX( err );
    return *this;
}

Vector& Vector::operator-=( const Vector& rhs ) {
    if( this->ptr() == rhs.ptr() ) {
        this->assign( 0 );
        return *this;
    }

    auto err = VecAXPY( this->ptr(), -1, rhs ); CHKERRXX( err );
    return *this;
}

Vector operator+( Vector lhs, Vector::scalar rhs ) {
    return lhs += rhs;
}

Vector operator-( Vector lhs, Vector::scalar rhs ) {
    return lhs -= rhs;
}

Vector operator*( Vector lhs, Vector::scalar rhs ) {
    return lhs *= rhs;
}

Vector operator/( Vector lhs, Vector::scalar rhs ) {
    return lhs /= rhs;
}

Vector operator+( Vector lhs, const Vector& rhs ) {
    return lhs += rhs;
}

Vector operator-( Vector lhs, const Vector& rhs ) {
    return lhs -= rhs;
}

Vector::scalar operator*( const Vector& lhs, const Vector& rhs ) {
    assert( lhs.size() == rhs.size() );

    return dot( lhs, rhs );
}

bool Vector::operator==( const Vector& other ) const {
    PetscBool eq;
    VecEqual( this->ptr(), other, &eq );
    return eq;
}

bool Vector::operator!=( const Vector& other ) const {
    return !( *this == other );
}

inline void Vector::set( const Vector::scalar* values,
        const Vector::size_type* indices,
        Vector::size_type size ) {

    /*
     * TODO: reimplement this smarter, i.e. use setvalues and local updates if
     * possible
     */

    auto err = VecSetValues( this->ptr(), size,
            indices, values, INSERT_VALUES ); CHKERRXX( err );

    err = VecAssemblyBegin( this->ptr() ); CHKERRXX( err );
    err = VecAssemblyEnd( this->ptr() ); CHKERRXX( err );
}

Vector::scalar dot( const Vector& lhs, const Vector& rhs ) {
    Vector::scalar x;
    VecDot( lhs, rhs, &x );
    return x;
}

Vector::scalar sum( const Vector& v ) {
    Vector::scalar x;
    auto err = VecSum( v, &x ); CHKERRXX( err );
    return x;
}

Vector::scalar max( const Vector& v ) {
    Vector::scalar x;
    auto err = VecMax( v, NULL, &x ); CHKERRXX( err );
    return x;
}

Vector::scalar min( const Vector& v ) {
    Vector::scalar x;
    auto err = VecMin( v, NULL, &x ); CHKERRXX( err );
    return x;
}

}
}
