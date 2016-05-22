/** **************************************************************************
 * Copyright (C) 2016 Adrien Devresse
 *
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301, USA.
 ** ***************************************************************************/
#ifndef MPI_TCC
#define MPI_TCC

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <cerrno>
#include <iostream>

#include <boost/atomic.hpp>
#include <boost/array.hpp>

#include "../mpi.hpp"
#include "../mpi_exception.hpp"


#include "mpi_type_mapper.hpp"

namespace mpi{

namespace impl{


inline std::string thread_opt_string(int opt_string){
    const boost::array<std::string,4> val_str = { { "MPI_THREAD_SINGLE", "MPI_THREAD_FUNNELED", "MPI_THREAD_SERIALIZED",
                              "MPI_THREAD_MULTIPLE" } };
    const boost::array<int, 4>  val = { { MPI_THREAD_SINGLE , MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED , MPI_THREAD_MULTIPLE } };

    for(std::size_t i=0; i < val.size(); ++i){
        if(val[i] == opt_string){
            return val_str[i];
        }
    }
    return "UNKNOWN";
}




template<typename T>
inline void _mpi_reduce_mapper(const T * ivalue, std::size_t n_value, MPI_Datatype data_type,
                            MPI_Op operation, const MPI_Comm comm,
                            T* ovalue){
    if(  MPI_Allreduce(static_cast<const void *>(ivalue),
                                    static_cast<void*>(ovalue),
                                    n_value, data_type, operation, comm) != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_Allreduce() ");
    }
}




} //impl



mpi_scope_env::mpi_scope_env(int *argc, char ***argv) : initialized(false){
    MPI_Initialized(&initialized);

    if(!initialized){
        const int thread_support= MPI_THREAD_MULTIPLE;
        int provided;
        if( MPI_Init_thread(argc, argv, thread_support, &provided) != MPI_SUCCESS ){
            throw mpi_exception(EINVAL, std::string("Unable to init MPI with ") + impl::thread_opt_string(thread_support));
        }
        if(provided != thread_support){
            std::cerr << "mpi_scope_env(MPI_Init_thread): MPI Thread level provided (" << impl::thread_opt_string(provided)
                         << ") different of required (" << impl::thread_opt_string(thread_support) << ")\n";
        }
    }
}


mpi_scope_env::~mpi_scope_env(){
    // Finalize only if initialized in this scope
    if(!initialized){
        MPI_Finalize();
    }
}


inline mpi_comm::message_handle::message_handle() :
   _status(),
   _msg()
{
   _status.MPI_SOURCE = _status.MPI_TAG = _status.MPI_ERROR = -1;
}


inline int mpi_comm::message_handle::tag() const{
                return _status.MPI_TAG;
}


inline int mpi_comm::message_handle::rank() const{
                return _status.MPI_SOURCE;
}


template<typename T>
inline std::size_t mpi_comm::message_handle::count() const{
    int count=0;
    if( MPI_Get_count(&_status, impl::_mpi_datatype_mapper(T()), &count) != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_Get_count() in message_handle ");
    }
    return static_cast<std::size_t>(count);
}



mpi_comm::mpi_comm() :
    _rank(0),
    _size(0),
    _comm(MPI_COMM_WORLD)
{

    MPI_Comm_rank(_comm, &_rank);
    MPI_Comm_size(_comm, &_size);

}



inline mpi_comm::~mpi_comm(){
    //
}





// MPI All Reduce wrappers

template <typename T>
inline T mpi_comm::all_max(const T & elems){
    T res;
    all_max(&elems, 1, &res);
    return res;
}

template <typename T>
inline void mpi_comm::all_max(const T * ivalue, std::size_t n_value, T* ovalue){
    impl::_mpi_reduce_mapper(ivalue, n_value, impl::_mpi_datatype_mapper(*ivalue),
                                    MPI_MAX, _comm,
                                    ovalue);
}


template <typename T>
inline T mpi_comm::all_sum(const T & pvalue){
    T res;
    all_sum(&pvalue, 1, &res);
    return res;
}

template <typename T>
inline void mpi_comm::all_sum(const T * ivalue, std::size_t n_value, T* ovalue){
    impl::_mpi_reduce_mapper(ivalue, n_value, impl::_mpi_datatype_mapper(*ivalue),
                                    MPI_SUM, _comm,
                                    ovalue);
}


// all_gather / all_gatherv
template <typename T, typename Y>
inline void mpi_comm::all_gather(const T & ivalues, std::vector<Y> & ovalues){

    impl::_mpi_flaterize<T> flatter(const_cast<T&>(ivalues));

    if(flatter.is_static_size()){
        ovalues.resize(flatter.get_flat_size() * size());
        all_gather(flatter.flat(), flatter.get_flat_size(), &(ovalues[0]));
    }else{
        // vlen arrays ( Allgatherv scenario )
        // do a Allgather before for length and then AllGatherv
        std::vector<std::size_t> elems_per_node;
        all_gather(flatter.get_flat_size(), elems_per_node);

        // Allgatherv
        ovalues.resize(std::accumulate(elems_per_node.begin(), elems_per_node.end(), 0));
        assert( ovalues.size() >= flatter.get_flat_size());
        all_gather(flatter.flat(), flatter.get_flat_size(), &(ovalues[0]), &(elems_per_node[0]));
    }
}


// all_gather
template <typename T, typename Y>
inline void mpi_comm::all_gather(const T* ivalues, std::size_t nvalues, Y* ovalues){

    if( MPI_Allgather(static_cast<const void*>(ivalues), nvalues, impl::_mpi_datatype_mapper(*ivalues),
                  static_cast<void*>(ovalues), nvalues, impl::_mpi_datatype_mapper(*ovalues), _comm) != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_Allgather() ");
    }
}

// all_gatherv
template <typename T, typename Y>
inline void mpi_comm::all_gather(const T* ivalues, std::size_t nvalues,
                                 Y* ovalues, const std::size_t * ovalues_per_node){

    std::vector<int> offsets(size(), 0), ovalues_per_node_int(ovalues_per_node, ovalues_per_node+ size());
    for(std::size_t i =1; i < offsets.size(); ++i){
        offsets[i] = offsets[i-1] + ovalues_per_node[i-1];
    }

    if( MPI_Allgatherv(static_cast<const void*>(ivalues), nvalues, impl::_mpi_datatype_mapper(*ivalues),
                  static_cast<void*>(ovalues), &(ovalues_per_node_int[0]), &(offsets[0]),
                  impl::_mpi_datatype_mapper(*ovalues), _comm) != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_Allgatherv() ");
    }
}





template<typename T>
inline void mpi_comm::broadcast(T* value, std::size_t nvalues, int root){

    if( MPI_Bcast(static_cast<void*>(value), static_cast<int>(nvalues),
                  impl::_mpi_datatype_mapper(*value), root, _comm) != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_Bcast() ");
    }
}

template<typename T>
inline void mpi_comm::broadcast( T & values, int root){
    impl::_mpi_flaterize<T> flatter(values);

    std::size_t vec_size;

    if(flatter.is_static_size() == false){
        // if our data container has a runtime variable length
        // broadcast first the length and then the data itself
        if(rank() == root){
            vec_size = flatter.get_flat_size();
        }
        broadcast<std::size_t>(&vec_size, 1, root);
    }else{
        vec_size = flatter.get_flat_size();
    }

    flatter.resize(vec_size);

    broadcast<typename impl::_mpi_flaterize<T>::base_type>(flatter.flat(), vec_size, root);
}




template <typename T>
inline void mpi_comm::send(const T & local_value, int dest_node, int tag){
    impl::_mpi_flaterize<T> flat_aspect(const_cast<T&>(local_value));

    send(flat_aspect.flat(), flat_aspect.get_flat_size(), dest_node, tag);
}


template <typename T>
inline void mpi_comm::send(const T * value, std::size_t n_value ,
                 int dest_node, int tag){

    if( MPI_Send(static_cast<void*>(const_cast<T*>(value)), static_cast<int>(n_value),
                   impl::_mpi_datatype_mapper(*value),
                  dest_node, tag, _comm) != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_send() ");
    }

}


inline mpi_comm::message_handle mpi_comm::probe(int src_node, int tag){
    mpi_comm::message_handle handle;

    if( MPI_Mprobe(src_node, tag, _comm, &(handle._msg), &(handle._status)) != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_Mprobe() ");
    }
    return handle;
}

template <typename T>
inline void mpi_comm::recv(int src_node, int tag, T & value){
    impl::_mpi_flaterize<T> flat_aspect(value);

    if(flat_aspect.is_static_size()){
        recv(src_node, tag, flat_aspect.flat(), flat_aspect.get_flat_size());
    }else{

    }
}

template <typename T>
inline void mpi_comm::recv(const mpi_comm::message_handle &handle, T & value){
    MPI_Status status;
    impl::_mpi_flaterize<T> flat_aspect(value);

    flat_aspect.resize(handle.count<typename impl::_mpi_flaterize<T>::base_type>());

    if( MPI_Mrecv(static_cast<void*>(flat_aspect.flat()), handle.count<typename impl::_mpi_flaterize<T>::base_type>(),
                 impl::_mpi_datatype_mapper(*(flat_aspect.flat())),
                 const_cast<MPI_Message*>(&(handle._msg)), &status)  != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_recv() ");
    }
}


template <typename T>
inline void mpi_comm::recv(int src_node, int tag, T* value, std::size_t n_value){
    MPI_Status status;

    if( MPI_Recv(static_cast<void*>(value), n_value, impl::_mpi_datatype_mapper(*value), src_node, tag, _comm, &status)  != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_recv() ");
    }
}


inline void mpi_comm::barrier(){
    if(MPI_Barrier(_comm) != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_barrier() ");
    }
}


} // mpi



#endif



