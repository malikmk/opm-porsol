#ifndef OPM_PETSCSOLVER_IMPL
#define OPM_PETSCSOLVER_IMPL

#include <ostream>

#include <opm/core/linalg/petscvector.hpp>
#include <iostream>

/*
 * This file holds the implementations of the template functions and methods
 * defined in petscsolver.hpp
 */

namespace Opm {
namespace Petsc {

template< typename T >
bool explicit_boolean_test( const Solver::Convergence_report< T >& );

template< typename T >
Solver::Convergence_report< T >::Convergence_report( T r, const char* d ) :
    reason( r ),
    description( d )
{}

template< typename T >
bool explicit_boolean_test( const Solver::Convergence_report< T >& x ) {
    return x.reason > 0;
}

template< typename T >
Solver::Convergence_report< T >::operator T() const {
    return this->reason;
}

template< typename T >
Solver::Convergence_report< T >::operator const char*() const {
    return this->description;
}

template< typename T >
Solver& Solver::set( T& head ) {
    return this->set( static_cast< const T& >( head ) );
}

template< typename T, typename... Args >
Solver& Solver::set( T&& head, Args&&... tail ) {
    return this->set( std::forward< T >( head ) )
        .set( std::forward< Args >( tail )... );
}

template< typename... Args >
Vector solve( const Matrix& A, const Vector& b, Args&& ... args ) {
    return Solver().set( std::forward< Args >( args )... )( A, b );
}

template< typename... Args >
Vector solve( const Vector& b, Args&& ... args ) {
    return Solver().set( std::forward< Args >( args )... )( b );
}

}
}

template< typename T >
std::ostream& operator<<( std::ostream& stream,
        const Opm::Petsc::Solver::Convergence_report< T >& report ) {
    return stream << static_cast< const char* >( report );
}

#endif //OPM_PETSCSOLVER_IMPL
