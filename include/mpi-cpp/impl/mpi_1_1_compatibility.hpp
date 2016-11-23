#ifndef _MPI_1_1_COMPATIBILITY_HPP
#define _MPI_1_1_COMPATIBILITY_HPP

#include <mpi.h>

//
// dummy implementation of MPI 3.0 minimal functionalities for mpi cpp
// provide compatibility for old MPI 1.1 API ( like the one of the BlueGene/Q )
//
// WARNING: the MPI_Improbe / MPI_Mprobe / MPI_Mrecv / MPI_Imrecv dummy implementaton DO NOT
//   are NOT thread safe. Do not use them in parallel threads unless you know what you are doing
//
namespace mpi {

	
// dummy MPI_message implementation for MPI < 3.0
struct MPI_Message{
	MPI_Message() : __tag(0), __source(0), __comm() {}
	// internal usage
	int __tag, __source;
	MPI_Comm __comm;
};


// unsafe MProbe
inline int MPI_Mprobe(int source, int tag, MPI_Comm comm,
    MPI_Message *message, MPI_Status *status){
	message->__tag = tag;
	message->__comm = comm;
	message->__source = source;
	return MPI_Probe(source, tag, comm, status);
}

// unsafe Improbe
int MPI_Improbe(int source, int tag, MPI_Comm comm,
    int *flag, MPI_Message *message, MPI_Status *status){
    message->__tag = tag;
    message->__comm = comm;
    message->__source = source;
	return MPI_Iprobe(source, tag, comm, flag, status);
}


// unsafe Imrecv
inline int MPI_Imrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message, MPI_Request *request){
	return MPI_Irecv(buf, count, datatype, message->__tag, message->__source, message->__comm, request);
}

// unsafe Mrecv
inline int MPI_Mrecv(void *buf, int count, MPI_Datatype datatype,
    MPI_Message *message, MPI_Status *status){
	return MPI_Recv(buf, count, datatype,
    message->__source, message->__tag, message->__comm, status);
}



}


#endif




