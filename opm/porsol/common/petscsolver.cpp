#include <opm/core/linalg/petsc.hpp>
#include <opm/core/linalg/petscmatrix.hpp>
#include <opm/core/linalg/petscsolver.hpp>
#include <opm/core/linalg/petscvector.hpp>
#include <opm/core/utility/ErrorMacros.hpp>

namespace Opm {
namespace Petsc {

static inline SNES default_Solver() {
    SNES x;
    SNESCreate( Petsc::comm(), &x );
    SNESSetFromOptions( x );

    return x;
}

Solver::Solver() : uptr< SNES >( default_Solver() ) {}

template<>
Solver::Convergence_report< KSPConvergedReason > Solver::converged() const {
    KSPConvergedReason x;
    KSPGetConvergedReason( *this, &x );
    return Convergence_report< KSPConvergedReason >( x, KSPConvergedReasons[ x ] );
}

template<>
Solver::Convergence_report< SNESConvergedReason > Solver::converged() const {
    SNESConvergedReason x;
    SNESGetConvergedReason( *this, &x );
    return Convergence_report< SNESConvergedReason >( x, SNESConvergedReasons[ x ] );
}

Vector Solver::operator()( const Matrix& A, const Vector& b ) {

    Vec raw_x;
    auto err = VecDuplicate( b, &raw_x );
    CHKERRXX( err );

    Vector x( raw_x );

    /* The PC Matrix defaults to the system Matrix */
    Mat pc_operator = A;
    PetscBool pcop_set;

    err = KSPGetOperatorsSet( *this, NULL, &pcop_set ); CHKERRXX( err );
    if( pcop_set ) {
        /* if the preconditioner operator has been set from elsewhere - make
         * sure we don't overwrite it.
         * 
         * We do as suggested by PETSc documentation,
         * http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPSetOperators.html
         */
        err = KSPGetOperators( *this, NULL, &pc_operator );
        CHKERRXX( err );
    }

    err = KSPSetOperators( *this, A, pc_operator ); CHKERRXX( err );

    err = KSPSolve( *this, b, x ); CHKERRXX( err );

    if( this->converged< KSPConvergedReason >() ) return x;

    OPM_THROW( std::runtime_error, this->converged< KSPConvergedReason >() );
}

Solver::operator PC() const {
    PC x;
    KSPGetPC( *this, &x );
    return x;
}

Solver::operator KSP() const {
    KSP x;
    SNESGetKSP( this->ptr(), &x );
    return x;
}

Solver& Solver::set() {
    return *this;
}

Solver& Solver::set( Pc_type type ) {
    auto err = PCSetType( *this, type ); CHKERRXX( err );
    return *this;
}

Solver& Solver::set( Ksp_type type ) {
    auto err = KSPSetType( *this, type ); CHKERRXX( err );
    return *this;
}

Solver& Solver::set( Snes_type type ) {
    auto err = SNESSetType( *this, type ); CHKERRXX( err );
    return *this;
}

Solver& Solver::set( const Matrix& B ) {
    Mat ksp_operator;
    auto err = KSPGetOperators( *this, &ksp_operator, NULL ); CHKERRXX( err );
    err = PetscObjectReference( (PetscObject)ksp_operator ); CHKERRXX( err );
    err = KSPSetOperators( *this, ksp_operator, B ); CHKERRXX( err );
    return *this;
}

Solver& Solver::set( const Solver::Linear_tolerance& tol ) {
    auto err = KSPSetTolerances( *this,
            tol.relative_tolerance,
            tol.absolute_tolerance,
            tol.divergence_tolerance,
            tol.maximum_iterations );
    CHKERRXX( err );
    return *this;
}

Solver::Linear_tolerance::Linear_tolerance() :
    relative_tolerance( PETSC_DEFAULT ),
    absolute_tolerance( PETSC_DEFAULT ),
    divergence_tolerance( PETSC_DEFAULT ),
    maximum_iterations( PETSC_DEFAULT )
{}

Solver::Linear_tolerance::Linear_tolerance(
        Solver::real rtol,
        Solver::real atol,
        Solver::real dtol,
        Solver::size_type maxit ) :
    relative_tolerance( rtol ),
    absolute_tolerance( atol ),
    divergence_tolerance( dtol ),
    maximum_iterations( maxit )
{}

Solver::Nonlinear_tolerance::Nonlinear_tolerance() :
    relative_tolerance( PETSC_DEFAULT ),
    absolute_tolerance( PETSC_DEFAULT ),
    solution_change_tolerance( PETSC_DEFAULT ),
    maximum_iterations( PETSC_DEFAULT ),
    maximum_function_evals( PETSC_DEFAULT )
{}

Solver::Nonlinear_tolerance::Nonlinear_tolerance(
        Solver::real rtol,
        Solver::real atol,
        Solver::real stol,
        Solver::size_type maxit,
        Solver::size_type maxf ) :
    relative_tolerance( rtol ),
    absolute_tolerance( atol ),
    solution_change_tolerance( stol ),
    maximum_iterations( maxit ),
    maximum_function_evals( maxf )
{}

}
}
