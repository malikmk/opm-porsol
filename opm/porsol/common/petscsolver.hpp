#ifndef OPM_PETSCSOLVER_H
#define OPM_PETSCSOLVER_H

#include <opm/core/linalg/petsc.hpp>
#include <opm/core/linalg/petscmixins.hpp>

namespace Opm {
namespace Petsc {

class Vector;
class Matrix;

template<>
struct deleter< _p_SNES >
{ void operator()( SNES x ) { SNESDestroy( &x ); } };

/// @brief Unified handle for PETSc preconditioner, krylov subspace methods
///     and nonlinear solvers.
///
///     PETSc solvers are arranged in a hierarchy, where krylov subspace method
///     solvers (from now on ksp or KSP) uses preconditioners (pc or PC), and
///     ksp is again used by non-linear solvers (snes or SNES).
///
///     Two interfaces are offered: a plain use-once function that constructs
///     solver, sets the parameters, solves the problem and returns the
///     solution vector and a context handle object interface that breaks up
///     this process into multiple steps and gives you slightly more control
///     over component or context reuse. In most cases you should prefer the
///     function call over reusing context, but for certain applications this
///     is infeasible.
///
///     There are two general uses:
///         #1: designing a new problem solution
///         #2: setting "permanent" parameters for the solvers once the optimal
///             solution has been found through profiling and testing.
///
///     For case #1 you generally only want the default constructor or the
///     plain functional interface. Consider solving the linear system Ax = b:
///     \code{.cpp}
///     auto x = Opm::petsc::solve( A, b );
///     \endcode
///     Various configurations are now just fetched from the PETSc option
///     database, which in turn is built from command line options, which
///     should be handled from argv to the Opm::petsc::petsc class. The
///     benefit of developing this way is that it is quick and easy to change
///     various parameters such as ksp algorithm, Matrix layout and tolerances.
///     Should you for some reason perfer using a solver context handle,
///     construct the solver and use operator() once all options has been set.
///
///     \code{.cpp}
///     Opm::petsc::solver handle;
///     /* set options */
///     auto x = handle( A, b );
///     \endcode
///     Again, when developing your algorithm I recommend setting as few
///     options as possible directly on the handle, but rather through command
///     line options.
///
///     For case #2, when the optimal parameters have been determined, you
///     should consider setting (and documenting!) the ideal options for the
///     current problem. While not necessary, it enables you and others to use
///     the options database for tuning different problems without interfering
///     with your configuration, as well as providing a form of documentation
///     and experience for the problem at hand. To set various parameters, use
///     the solver::set() family of methods (if using the context handle) or by
///     adding the parameters to your call to solve.
///
///     All options can be set with the solver::set() method. This is
///     overloaded based on the -type- of the parameter, and not by name, and
///     repeated calls can be chained. This is easier explained with example
///     code:
///     \code{.cpp}
///     Opm::Petsc::Solver::Ksp_type linalg( "cg" ); // conjugate gradient
///     Opm::Petsc::Solver::Pc_type precond( "ilu" ); // ILU(0) preconditioner
///     auto op = get_precond_operator();
///
///     // functional:
///     auto xf = Opm::Petsc::solve( A, b, linalg, precond, op );
///
///     // call-on-object:
///     Opm::Petsc::Solver handle;
///     handle.set( linalg ).
///            set( precond ).
///            set( op );
///
///     auto xo = handle( A, b );
///
///     // idiomatically, with args constructed as they are passed:
///     auto xi = Opm::Petsc::solve( A, b,
///         Opm::Petsc::Solver::Ksp_type( "gmres" ),
///         Opm::Petsc::Solver::Pc_type( "sor" ) );
///     \endcode
///
///     Please note that the order of calls to set or the order of arguments,
///     except for A and b, is irrelevant. Both these solutions are equivalent.
///
///     The operator() was chosen so that you can do a "partial application" of
///     options determined from somewhere else, and then return the correct
///     "function" to call. Again, illustrated by a code example:
///
///     \code{.cpp}
///     Opm::Petsc::Solver determine_ideal_algorithm() {
///         Opm::Petsc::Solver handle;
///         /* black magic that determines if gmres or cg is the best */
///         switch( rand() % 2 ) {
///             case 0:
///                 return handle.set( Solver::Ksp_type( "cg" ) );
///             case 1:
///                 return handle.set( Solver::Ksp_type( "gmres" ) );
///         }
///
///         return handle;
///     }
///
///     int fun() {
///         auto A = get_Matrix();
///         auto b = get_vec();
///
///         auto linsolve = determine_ideal_algorithm();
///         auto x = linsolve( A, b );
///     }
///     \endcode
///
///     On behaviour: By default, the operator/Matrix for the preconditioner is
///     A, unless some other operator has been given.
///
///     The solver handle is implicitly convertible to the PETSc types SNES,
///     KSP and PC, so you can directly call PETSc functions with this object.
///     I do, however, not recommend this in production code - rather, if it is
///     an option that needs setting it should be implemented through here.

class Solver : public uptr< SNES > {
    public:
        struct Linear_tolerance;
        struct Nonlinear_tolerance;

        template< typename T >
        struct Convergence_report;

        /// @brief Alias for PetscReal, typically a (long) double
        typedef PetscReal real;
        /// @brief Alias for PetscScalar, typically a (long) double
        typedef PetscScalar scalar;
        /// @brief Alias for PetscInt, typically a long integer
        typedef PetscInt size_type;

        /// @brief Constructor
        Solver();

        /*
         * Make type aliases for the algorithm name strings
         * This is used to explicitly set what algorithm to use for various
         * parts of the numerical process (preconditioning, linear and
         * non-linear part, respectively)
         *
         * This macro is defined in petscmixins.hpp
         */
        /// @brief  Type alias for PCType, typically const char*
        ///         The pc_type alias is used to set preconditioner algorithm
        ///         in the solver.
        mknewtype( Pc_type, PCType );
        /// @brief  Type alias for KSPType, typically const char*
        ///         The ksp_type alias is used to set linear solver (krylov
        ///         subspace method) algorithm in the solver.
        mknewtype( Ksp_type, KSPType );
        /// @brief  Type alias for SNESType, typically const char*
        ///         The snes_type alias is used to set the nonlinear algorithm
        ///         in the solver.
        mknewtype( Snes_type, SNESType );

        operator PC() const;
        operator KSP() const;

        Solver& set( const Linear_tolerance& );
        Solver& set( const Nonlinear_tolerance& );

        /// @brief Set operator to use for the preconditioner
        Solver& set( const Matrix& );

        /*
         * petsc uses strings to set algorithm types. This was probably
         * chosen because petsc supports ad-hoc adding new algorithms or
         * implementations, which means you need a dynamic system for
         * registering and addressing, and enums (which would've otherwise
         * been the superiour choice) does not offer that. We could
         * implement our own enums of algorithms we choose to support, but
         * just using strings is fine too. However, in order to provide at
         * least a bit of safety, we introduce new thin types over the
         * different -kinds- of algorithms. To set, say, the conjugent
         * gradient method, call solver.set( ksp_type( "cg" ) );
         *
         * An added benefit is that ksp_type( "cg" ); communicates intent a
         * lot better than const char* = "cg";
         */
        /// @brief Set preconditioner algorithm
        Solver& set( Pc_type );
        /// @brief Set ksp algorithm
        Solver& set( Ksp_type );
        /// @brief Set non-linear algorithm
        Solver& set( Snes_type );

        //TODO: support initial guess by passing a vector&& which provides
        //storage & gets returned by operator()

        /*
         * This trick to converts T& into const T& if it cannot find a suitable
         * T& overload. Since the variadic set() perfectly forwards its
         * arguments, we need a way to convert from T& to const T&.
         *
         * The problem occurs when, which is common, an object is declared as
         * non-const in the enclosing scope. This is passed as T& to set, which
         * is distinct from set( const T& ). This would again call the
         * templated T&, leading to infinite recursion. Unless a suitable
         * non-templated set( T& ) is defined, this one is called which
         * attempts to find a const'd method.
         *
         */
        template< typename T >
        Solver& set( T& );

        template< typename T, typename... Args >
        Solver& set( T&&, Args&& ... );

        /// @brief Solve the linear system Ax = b
        /// \param[in]  A   The system Matrix
        /// \param[in]  b   The system solution
        /// \return     x   The augmenting vector
        Vector operator()( const Matrix& A, const Vector& b );

    private:
        /* 
         * base case for the variadic template params setter, this is a no-op.
         * Should not be available to end-users, but must be available to
         * solve().
         */
        Solver& set();

        template< typename... Args >
        friend Vector solve( const Matrix&, const Vector&, Args&& ... );

        template< typename T >
        Convergence_report< T > converged() const;
};

/// @brief  Solve the linear system Ax = b
/// \param[in]  A       The system Matrix
/// \param[in]  b       The system solution
/// \param[in] args...  An arbitrary (possibly none) number of options
/// \return     x       The augmenting vector
template< typename... Args >
Vector solve( const Matrix&, const Vector&, Args&& ... );

/// @brief  Tolerance parameters for linear equations.
///         A simple collection of relative tolerance, absolute tolerance,
///         divergence tolerance and maximum iterations.
struct Solver::Linear_tolerance {
    Linear_tolerance();
    Linear_tolerance( real, real, real, size_type );

    Solver::real relative_tolerance;
    Solver::real absolute_tolerance;
    Solver::real divergence_tolerance;
    Solver::size_type maximum_iterations;
};

/// @brief  Tolerance parameters for non-linear equations.
///         A simple collection of relative tolerance, absolute tolerance,
///         tolerance of the norm change between steps, maximum iterations and
///         maximum function evaluations.
struct Solver::Nonlinear_tolerance {
    Nonlinear_tolerance();
    Nonlinear_tolerance( real, real, real, size_type, size_type );

    Solver::real relative_tolerance;
    Solver::real absolute_tolerance;
    Solver::real solution_change_tolerance;
    Solver::size_type maximum_iterations;
    Solver::size_type maximum_function_evals;
};

/// @brief  Solution convergence report.
///         This simple struct reports whether or not the attempted solution
///         converged. Is implicitly convertible to bool, so convergence can be
///         tested with if( report ). Has std::ostream<< overload provided, but
///         can be accessed directly if you want to generate your own messages
///         log entries.
template< typename T >
struct Solver::Convergence_report :
    public explicit_bool_conversion< Solver::Convergence_report< T > > {
    // TODO: add iteration number? remove implicit T()-conversion?

    T reason;
    const char* description;

    /* we want the implicit conversions, but we don't want (especially the
     * bool case) to break type safety. Post C++11 this can be handled by
     * marking the operator bool() explicit, but GCC only supports that
     * after 4.5. A solution is to explicitly implement comparison
     * operators for the convergence_report that will fail on anything but
     * the desired case: if( converged ) {}
     *
     * For a reference on this, see:
     * http://en.wikibooks.org/wiki/More_C++_Idioms/Safe_bool
     */

    Convergence_report( T, const char* );

    operator T() const;
    operator const char*() const;
};

}
}

#include <opm/core/linalg/petscsolver_impl.hpp>

#endif //OPM_PETSCSOLVER_H
