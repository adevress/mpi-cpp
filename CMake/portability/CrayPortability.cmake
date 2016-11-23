##
## Portability check on Blue Gene Q environment

include(CompilerFlagsHelpers)

if(IS_DIRECTORY "/opt/cray" AND (NOT DEFINED IS_CRAY) ) 
    set(IS_CRAY TRUE)
    message(STATUS "Cray supercompter platform detected, Enable Cray specific configuration")
endif()


if(IS_CRAY)

	##  The Cray Driver (compiler wrapper) automatically map mpi 
        ##  Trying to detect and use the MPI library will only lead to conflict with the cray library
	set(MPI_LIBRARIES "")
	set(MPI_C_LIBRARIES "")	
	set(MPI_CXX_LIBRARIES "")

else()

endif()



