#include <opm/core/linalg/petsc.hpp>

namespace Opm {
namespace Petsc {

Petsc::Petsc(
        int* argc,
        char*** argv,
        const char* file,
        const char* help ) {

    PetscInitialize( argc, argv, file, help );
}

Petsc::~Petsc() {
    PetscFinalize();
}

MPI_Comm Petsc::comm() {
    /* Slight hack to get things working. This sets the communicator used when
     * constructing new objects - however, this should be configurable. When full
     * petsc support is figured out this will be modifiable, possibly through
     * some configuration option or a set_comm function.
     */
    return PETSC_COMM_WORLD;
}

}
}
