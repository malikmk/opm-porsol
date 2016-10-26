#ifndef OPM_PETSC_MIXINS_H
#define OPM_PETSC_MIXINS_H

#include <memory>

namespace Opm {
namespace Petsc {

/*
 * Deleters are implemented in the same files that declares the class they're
 * to delete, i.e. petscvector.hpp for vector deleter.
 */
template< typename T > struct deleter;

/*
 * petsc typedefs its handles from _p_Vec* (etc) to Vec. It is certainly nicer
 * to, in the class declaration, declare the type you -actually- want exposed,
 * so we use a template trick to remove the pointer part of the type and infer
 * the underlying type. This is done by first declaring a shell template class,
 * to introduce the symbol, then specialise for a pointer type. Notice that
 * uptr is only forward declared and never gets an implementation - this is by
 * design, so that we get compiler errors in case it is used wrong.
 */
template< typename T > class uptr;

/*
 * Here we take the pointer typedef, e.g. Vec and Mat, and strip the pointer so
 * that unique_ptr can use them. This avoids double indirection, making sure
 * the uptr mixin provides no extra overhead.
 */
template< typename T >
class uptr< T* > : private std::unique_ptr< T, deleter< T > > {
    private:
        typedef T* pointer;
        typedef std::unique_ptr< T, deleter< T > > base;

    public:
        /*
         * We only want a specific subset of constructors, and generally not
         * other methods.  The reasoning here is that avoiding direct pointer
         * access makes code easier to change in the future and less reliant on
         * actual implementation.
         */

        uptr< T* >( uptr< T* >&& );
        uptr< T* >( T* );

        operator pointer() const;

    protected:
        inline pointer ptr() const;

        void swap( uptr& );
};

/*
 * GCC4.4 does not support explicit conversion operators, so we implement the
 * feature manually for the cases that might need them (see: petscsolver.cpp
 * and convergence_report.
 *
 * In order to implement explicit operator bool() support, public inherit from
 * this and imlpement the function explicit_boolean_test( const your_type& ):
 *
 * class new_type : explicit_bool_conversion< new_type > {};
 * explicit_boolean_test( const new_type& x ) { return x.is_true(); }
 *
 * Obviously, explicit_boolean_test must be a visible symbol.
 */
class bool_base {
    public:
        operator bool() const = delete;
        void unsupported_comparison_between_types() const {};

    protected:
        bool_base() = default;
        bool_base( const bool_base& ) = default;
        bool_base& operator=( const bool_base& ) = default;
};

template< typename T >
class explicit_bool_conversion : private bool_base {
    public:
        operator bool() const {
            return explicit_boolean_test( static_cast< const T& >( *this ) ) ?
                &bool_base::unsupported_comparison_between_types : false;
        }
};

/*
 * Works like Haskell's newtype keyword, i.e. constructs a new type "alias".
 * This is particularly useful in combination with overloading and will in most
 * cases give simpler, cleaner and easier-to-read code.
 *
 * The newtype mixin creates a thin, typed layer (which compiles to zero
 * overhead). The obvious use case is to carry semantics along with data. e.g.
 * passing the newtype "age" instead of a raw int.
 *
 * For ease of implementation it only supports read-only data, i.e. once the
 * type is set it is not possible to update the data, as the only access to it
 * is implicit conversion. Meant for primary types.
 *
 * We offer the macro mknewtype( typename, basetype ) for ease of creation.
 * Since we want to use the constructor from newtype and C++11 support in
 * GCC4.4 haven't got inheriting constructors (using Base::Base;), we must
 * implement "all" constructors manually.
 */

template< typename T >
class newtype {
    public:
        template< typename U > newtype( U&& t );
        operator T() const;

    private:
        T data;
};

#define mknewtype( name, basetype )                         \
struct name : public Opm::Petsc::newtype< basetype > {      \
    template< typename T > name( T&& t ) :                  \
        Opm::Petsc::newtype< basetype >(                    \
                std::forward< T >( t ) ) {}                 \
}

/*
 * Implementations
 */

template< typename T>
uptr< T* >::uptr( uptr< T* >&& x ) : base( std::move( x ) ) {}

template< typename T>
uptr< T* >::uptr( T* x ) : base( x ) {}

template< typename T >
uptr< T* >::operator uptr< T* >::pointer() const {
    return this->ptr();
}

template< typename T >
T* uptr< T* >::ptr() const {
    return this->get();
}

template< typename T >
void uptr< T* >::swap( uptr< T* >& rhs ) {
    this->std::unique_ptr< T, deleter< T > >::swap( rhs );
}

template< typename T >
template< typename U >
newtype< T >::newtype( U&& x ) : data( std::forward< U >( x ) ) {}

template< typename T >
newtype< T >::operator T() const {
    return this->data;
}

template < typename T >
bool operator==( const explicit_bool_conversion< T >& lhs, bool b) {
    return b == static_cast< bool >( lhs );
}

template < typename T >
bool operator==( bool b, const explicit_bool_conversion< T >& rhs ) {
    return b == static_cast< bool >( rhs );
}

template < typename T, typename U >
bool operator==( const explicit_bool_conversion< T >& lhs,
        const explicit_bool_conversion< U >& rhs ) {

    lhs.unsupported_comparison_between_types();
    return false;
}

template < typename T, typename U >
bool operator!=( const explicit_bool_conversion< T >& lhs,
        const explicit_bool_conversion< U >& rhs ) {

    lhs.unsupported_comparison_between_types();
    return false;
}

}
}

#endif //OPM_PETSC_MIXINS_H
