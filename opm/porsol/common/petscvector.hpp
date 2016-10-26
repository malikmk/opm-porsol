#ifndef OPM_PETSCVECTOR_H
#define OPM_PETSCVECTOR_H

/*
 * C++ bindings to Petsc Vec.
 *
 * Provides an easy-to-use C++-esque implementation with no extra indirection
 * for Petsc Vec. Slightly radical in design, but in spirit with Petsc: no
 * direct memory access is allowed, and no iterators are provided. Due to the
 * nature of Petsc there usually is no guarantee that the memory you want to
 * access lives in your process. All interactions with the vector should happen
 * through functions - possibly general higher-order functions at a later time.
 *
 * The idea is to get (some) of Petsc's power, but easier to use. Tested to
 * without effort be able to use MPI. Supports arithmetic operators and should
 * be easy to reason about. Lifetime is managed through unique_ptr for safety
 * and simplicity - no need to implement custom assignment and move operators.
 *
 * Note that a default-constructed vector is NOT allowed. While this makes it
 * slightly harder to use in a class (you most likely shouldn't anyways), it
 * disallows more invalid states at compile-time.
 *
 * Also note that there are no set-methods for values. This is by design, but
 * they might be added later if there is a need for it. Structurally, this
 * leaves the vector immutable.
 *
 * Please note that all methods in the vector class only deal with it's
 * structure. This is by design - everything needed to compute something about
 * the vector is or will be provided as free functions, in order to keep
 * responsibilities separate.
 *
 * The vector gives no guarantees for internal representation.
 *
 * TODO: When full C++11 support can be used, most constructors can be
 * rewritten to use ineheriting constructors.
 */

#include <opm/core/linalg/petsc.hpp>
#include <opm/core/linalg/petscmixins.hpp>

#include <petscvec.h>

#include <vector>

namespace Opm {
namespace Petsc {

    template<>
    struct deleter< _p_Vec >
    { void operator()( Vec x ) { VecDestroy( &x ); } };

    /// @brief PETSc Vector object
    ///
    /// The Vector class mirrors some aspects of std::vector's interface, but
    /// the implementation can be fully distributed, so there is no direct
    /// memory or member access. The Vector may also be sparse.
    ///
    /// Vector is implicitly convertible to Vec, so it can be used directly in
    /// PETSc functions. However, I do not recommend this for other things than
    /// development and debugging - if some feature is needed from the Vector,
    /// consider implementing it in the library.

    class Vector : public uptr< Vec > {
        public:
            typedef PetscScalar scalar;
            typedef PetscInt size_type;

            using uptr< Vec >::operator=;

            Vector() = delete;

            /* This can in (full) C++11 be implemented with
             * using uptr< Vec >::uptr; i.e. inheriting constructors. GCC4.4
             * has no support for this, so we must implement the call
             * ourselves.
             */
            /// @brief Acquire ownership of a raw PETSc handle
            Vector( Vec );
            /// @brief Copy constructor.
            Vector( const Vector& );
            /// @brief Move constrcutor.
            Vector( Vector&& );

            /// @brief Constructor. Does not populate the Vector with values.
            /// \param[in] size     Number of elements
            explicit Vector( size_type );
            /// Constructor. Populate [0..n-1] with scalar. This is equivalent
            /// Vector v( size ); v.assign( scalar );
            /// \param[in]  size    Number of elements
            /// \param[in]  scalar  Value to set
            explicit Vector( size_type, scalar );

            /// @brief Construct from std::vector.
            /// \param[in] Vector   std::vector to copy from
            Vector( const std::vector< scalar >& );
            /// @brief Construct from std::vector at the indices provided by
            /// indexset. Negative indices are ignored.
            /// \param[in] Vector   std::vector to copy from
            /// \param[in] indexset indices to assign values from the Vector.
            Vector( const std::vector< scalar >&,
                    const std::vector< size_type >& indexset );

            /// @brief Get Vector size.
            /// \param[out] size    Number of elements in the Vector
            size_type size() const;

            /// @brief Assign a value to all elements in the Vector.
            /// \param[in] scalar  The value all elements will have
            void assign( scalar );

            /// @brief Add a value to all elements in the Vector.
            /// \param[in] scalar   Value to add to all elements
            Vector& operator+=( scalar );
            /// @brief Subtract a value from all elements in the Vector.
            /// \param[in] scalar   Value to subtract from all elements
            Vector& operator-=( scalar );
            /// @brief Scalar multiplication.
            /// \param[in] scalar   Value to scale with
            Vector& operator*=( scalar );
            /// @brief Inverse scalar multiplication (division)
            /// \param[in] scalar   Value to scale with
            Vector& operator/=( scalar );

            /// @brief Vector addition, x + y.
            /// \param[in] Vector   Vector to add
            Vector& operator+=( const Vector& );
            /// @brief Vector subtraction, x - y.
            /// \param[in] Vector   Vector to subtract
            Vector& operator-=( const Vector& );

            /* these currently do not have to be friends (since Vector is
             * implicitly convertible), but they are for least possible
             * surprise
             */
            friend Vector operator+( Vector, scalar );
            friend Vector operator-( Vector, scalar );
            friend Vector operator*( Vector, scalar );
            friend Vector operator/( Vector, scalar );

            friend Vector operator+( Vector, const Vector& );
            friend Vector operator-( Vector, const Vector& );
            friend scalar operator*( const Vector&, const Vector& );

            /// @brief Equality check.
            /// \param[out] eq  Returns true if equal, false if not
            bool operator==( const Vector& ) const;
            /// @brief Inequality check.
            /// \param[out] eq  Returns false if equal, true if not
            bool operator!=( const Vector& ) const;

        private:
            inline void set( const scalar*, const size_type*, size_type );
    };

    /// @brief Calculate the dot product of two Vectors
    Vector::scalar dot( const Vector&, const Vector& );

    /// @brief Calculate the sum of all values in the Vector
    Vector::scalar sum( const Vector& );
    /// @brief Find the biggest element in the Vector
    Vector::scalar max( const Vector& );
    /// @brief Find the smallest element in the Vector
    Vector::scalar min( const Vector& );

}
}

#endif //OPM_PETSCVECTOR_H
