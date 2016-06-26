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
#include <vector>
#include <cerrno>
#include <iostream>
#include <limits>

#include <unistd.h>

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


// Some MPI implementation do not respect the norm and accept only void* instead of const void*
// for some read only input parameter
// force conversion to avoid any issue
template<typename T>
inline void* _to_void_pointer(const T* orig_ptr){
	return static_cast<void*>(const_cast<T*>(orig_ptr));
}


inline void _check_mpi_result(int mpi_res, int err_code,
                      const std::string & err_msg){
    if(mpi_res != MPI_SUCCESS){
        throw mpi_exception(err_code, err_msg);
    }

}


template<typename T>
inline void _mpi_reduce_mapper(const T * ivalue, std::size_t n_value, MPI_Datatype data_type,
                            MPI_Op operation, const MPI_Comm comm,
                            T* ovalue){
    _check_mpi_result(
                MPI_Allreduce(impl::_to_void_pointer(ivalue),
                                    impl::_to_void_pointer(ovalue),
                                    n_value, data_type, operation, comm),
                EIO,
                "Error during MPI_Allreduce() ");
}


} //impl


mpi_scope_env::mpi_scope_env(int *argc, char ***argv) : initialized(false){
    MPI_Initialized(&initialized);

    if(!initialized){
        const int thread_support= MPI_THREAD_MULTIPLE;
        int provided;

        impl::_check_mpi_result(
                    MPI_Init_thread(argc, argv, thread_support, &provided),
                    EINVAL,
                    std::string("Unable to init MPI with ") + impl::thread_opt_string(thread_support));

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



inline void mpi_scope_env::enable_exception_report(){
    MPI_Comm_set_errhandler(MPI_COMM_WORLD,  MPI_ERRORS_RETURN);
}


template<typename Value>
class mpi_future_internal : boost::noncopyable {
public:

    inline mpi_future_internal(Value & v) :
        _v(v),
        _req(MPI_REQUEST_NULL),
        _status(),
        _flags(){}

    inline ~mpi_future_internal(){
        if( is_completed() == false){
           wait();
        }
    }

    inline void wait(){
        if(is_completed() == false){
            impl::_check_mpi_result(
                        MPI_Wait(&_req, MPI_STATUS_IGNORE),
                        EIO,
                        "Error during MPI_Wait() in message_handle "
            );
            set_completed();
        }
    }

    inline bool wait_for(std::size_t us_time){
        int flag =0;
        if(is_completed() == false){

            do{
                impl::_check_mpi_result(
                            MPI_Wait(&_req, MPI_STATUS_IGNORE),
                            EIO,
                            "Error during MPI_Wait() in message_handle "
                );

                if(flag){
                    set_completed();
                    return true;
                }
                if(us_time ==0){
                    return false;
                }

                us_time -= 1;
                usleep(1);
            } while(1);

        }
        return true;
    }


    inline Value get(){
        wait();
        return _v;
    }


    inline bool is_completed() const {
        return _flags[future_completed];
    }

    inline void set_completed(){
        _flags[future_completed] = true;
    }

    inline MPI_Request & get_ref_req(){
        return _req;
    }


    Value _v;
    MPI_Request _req;
    MPI_Status _status;

    std::bitset<8> _flags;

    enum future_flags{
        future_completed=0x1
    };
};

template<typename Value>
inline void check_intern_validity(const boost::shared_ptr<mpi_future_internal <Value> > & intern_ptr){
    if(!intern_ptr){
        throw mpi_invalid_future();
    }
}


template<typename Future>
static void manage_completed_single_handle(int index,
                                const std::vector<Future> mpi_futures,
                                Future & completed_future){
    if(index >= 0){
        assert(std::size_t(index) < mpi_futures.size());
        completed_future = mpi_futures[index];
        completed_future._intern->set_completed();
    }
}


template<typename Future>
static void manage_completed_multiple_handle(int outcount, std::vector<int> array_indices,
                                const std::vector<Future> mpi_futures,
                                std::vector<Future> & completed_future){
    if(outcount > 0){
        for(int i =0; i < outcount; ++i){
            const int index = array_indices[i];
            assert(std::size_t(index) < mpi_futures.size());

            completed_future.push_back( mpi_futures[index] );
            completed_future.back()._intern->set_completed();
        }
    }
}

template<typename Value>
inline mpi_future<Value>::mpi_future() :
    _intern()
{}


template<typename Value>
inline mpi_future<Value>::mpi_future(const mpi_future<Value> & other) :
    _intern(other._intern)
{

}

template<typename Value>
inline mpi_future<Value> & mpi_future<Value>::operator=(const mpi_future<Value> & other){
    if(this != &other){
        mpi_future<Value> tmp(other);
        this->swap(tmp);
    }
    return *this;
}



template<typename Value>
inline mpi_future<Value>::~mpi_future(){

}


template<typename Value>
inline Value mpi_future<Value>::get(){
    Value res = _intern->get();
    _intern.reset();
    return res;
}

template<typename Value>
inline void mpi_future<Value>::wait(){
    check_intern_validity(_intern);
    _intern->wait();
}

template<typename Value>
inline bool mpi_future<Value>::wait_for(std::size_t us_time){
    check_intern_validity(_intern);
    return _intern->wait_for(us_time);
}


template<typename Value>
inline MPI_Request & mpi_future<Value>::get_request(){
    return _intern->_req;
}




template<typename Value>
bool mpi_future<Value>::valid() const{
    return _intern;
}

template<typename Value>
bool mpi_future<Value>::completed() const{
    check_intern_validity(_intern);
    return _intern->is_completed();
}


template<typename Value>
void mpi_future<Value>::swap(mpi_future<Value> &other){
    _intern.swap(other._intern);
}

template<typename Value>
typename std::vector<mpi_future<Value> > mpi_future<Value>::wait_some_for(typename std::vector<mpi_future<Value> > & mpi_futures,
                                                                          std::size_t us_time){

    typename std::vector<mpi_future<Value> > res;

    const std::size_t n_reqs = mpi_futures.size();
    int outcount=0, incount = static_cast<int>(n_reqs);

    if(n_reqs == 0)
        return res;

    std::vector<MPI_Request> reqs(n_reqs);
    std::vector<int> array_indices(n_reqs, -1);
    std::vector<MPI_Status> array_status(n_reqs);

    for(std::size_t i =0; i < n_reqs; ++i){
        reqs[i] = mpi_futures[i].get_request();
    }

    if(us_time == std::numeric_limits<std::size_t>::max()){
        impl::_check_mpi_result(
                    MPI_Waitsome(incount, &(reqs[0]), &outcount, &(array_indices[0]), &(array_status[0])),
                    EIO,
                    "Error during MPI_Waitsome()");


    }else{
        do{
            impl::_check_mpi_result(
                        MPI_Testsome(incount, &(reqs[0]), &outcount, &(array_indices[0]), &(array_status[0])),
                        EIO,
                        "Error during MPI_Testsome()");

            if(outcount > 0 || us_time == 0)
                break;

            us_time -= 1;
            usleep(1);
        } while(1);
    }

    manage_completed_multiple_handle< mpi_future<Value> >(outcount, array_indices,
                             mpi_futures, res);
    return res;

}


template<typename Value>
typename std::vector<mpi_future<Value> > mpi_future<Value>::wait_some(typename std::vector<mpi_future<Value> > & mpi_futures){
    return wait_some_for(mpi_futures, std::numeric_limits<std::size_t>::max());
}



template<typename Value>
mpi_future<Value> mpi_future<Value>::wait_any_for(
        typename std::vector< mpi_future<Value> > & mpi_futures,
        std::size_t us_time){
    mpi_future<Value> res;

    const std::size_t n_reqs = mpi_futures.size();
    int incount = static_cast<int>(n_reqs);

    if(n_reqs == 0)
        return res;

    std::vector<MPI_Request> reqs(n_reqs);
    int index = -1, flag = 0;
    MPI_Status status;

    for(std::size_t i =0; i < n_reqs; ++i){
        reqs[i] = mpi_futures[i].get_request();
    }


    if(us_time == std::numeric_limits<std::size_t>::max()){
        impl::_check_mpi_result(
                    MPI_Waitany(incount, &(reqs[0]), &index, &(status)),
                    EIO,
                    "Error during MPI_Waitsome()");
    }else{
        do{
            impl::_check_mpi_result(
                        MPI_Testany(incount, &(reqs[0]), &index, &flag, &(status)),
                        EIO,
                        "Error during MPI_Testany()");

            if(index > 0 || us_time ==0){
                break;
            }

            us_time -= 1;
            usleep(1);

        } while(1);
    }

    manage_completed_single_handle(index, mpi_futures, res);
    return res;
}

template<typename Value>
mpi_future<Value> mpi_future<Value>::wait_any(
        typename std::vector< mpi_future<Value> > & mpi_futures){
    return wait_any_for(mpi_futures, std::numeric_limits<std::size_t>::max());
}


template<typename Value>
bool mpi_future_is_invalid(const mpi_future<Value> & future){
    return (!future->_intern);
}

template<typename Value>
void mpi_future<Value>::filter_invalid(typename std::vector<mpi_future<Value> > & mpi_futures){
    typename std::vector< mpi_future<Value> >::iterator end = std::remove_if(mpi_futures.begin(), mpi_futures.end(), mpi_future_is_invalid<Value>);
    mpi_futures.resize(std::distance(mpi_futures.begin(), end));
}

template<typename Value>
bool mpi_future_is_completed(const mpi_future<Value> & future){
    return (future.completed());
}

template<typename Value>
void mpi_future<Value>::filter_completed(typename std::vector<mpi_future<Value> > & mpi_futures){
    typename std::vector< mpi_future<Value> >::iterator end = std::remove_if(mpi_futures.begin(), mpi_futures.end(), mpi_future_is_completed<Value>);
    mpi_futures.resize(std::distance(mpi_futures.begin(), end));
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

inline bool mpi_comm::message_handle::is_valid() const{
    return (_status.MPI_SOURCE != -1);
}


template<typename T>
inline std::size_t mpi_comm::message_handle::count() const{
    int count=0;
    impl::_check_mpi_result(
        MPI_Get_count(&_status, impl::_mpi_datatype_mapper(T()), &count),
        EIO,
        "Error during MPI_Get_count() in message_handle "
    );
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
    using namespace impl;

    if( MPI_Allgather(_to_void_pointer(ivalues), nvalues, impl::_mpi_datatype_mapper(*ivalues),
                  _to_void_pointer(ovalues), nvalues, impl::_mpi_datatype_mapper(*ovalues), _comm) != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_Allgather() ");
    }
}

// all_gatherv
template <typename T, typename Y>
inline void mpi_comm::all_gather(const T* ivalues, std::size_t nvalues,
                                 Y* ovalues, const std::size_t * ovalues_per_node){
    using namespace impl;

    std::vector<int> offsets(size(), 0), ovalues_per_node_int(ovalues_per_node, ovalues_per_node+ size());
    for(std::size_t i =1; i < offsets.size(); ++i){
        offsets[i] = offsets[i-1] + ovalues_per_node[i-1];
    }

    if( MPI_Allgatherv(_to_void_pointer(ivalues), nvalues, _mpi_datatype_mapper(*ivalues),
                  _to_void_pointer(ovalues), &(ovalues_per_node_int[0]), &(offsets[0]),
                  _mpi_datatype_mapper(*ovalues), _comm) != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_Allgatherv() ");
    }
}





template<typename T>
inline void mpi_comm::broadcast(T* value, std::size_t nvalues, int root){
    using namespace impl;

    if( MPI_Bcast(_to_void_pointer(value), static_cast<int>(nvalues),
                  _mpi_datatype_mapper(*value), root, _comm) != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_Bcast() ");
    }
}

template<typename T>
inline void mpi_comm::broadcast( T & values, int root){
    using namespace impl;

    _mpi_flaterize<T> flatter(values);

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

    broadcast<typename _mpi_flaterize<T>::base_type>(flatter.flat(), vec_size, root);
}




template <typename T>
inline void mpi_comm::send(const T & local_value, int dest_node, int tag){
    impl::_mpi_flaterize<T> flat_aspect(const_cast<T&>(local_value));

    send(flat_aspect.flat(), flat_aspect.get_flat_size(), dest_node, tag);
}


template <typename T>
inline void mpi_comm::send(const T * value, std::size_t n_value ,
                 int dest_node, int tag){
    using namespace impl;

    if( MPI_Send(_to_void_pointer(const_cast<T*>(value)), static_cast<int>(n_value),
                   _mpi_datatype_mapper(*value),
                  dest_node, tag, _comm) != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_send() ");
    }

}


template <typename T>
inline mpi_future<T> mpi_comm::send_async(const T & local_value, int dest_node, int tag){
    using namespace impl;

    impl::_mpi_flaterize<T> flat_aspect(const_cast<T&>(local_value));

    mpi_future<T> fut;
    fut._intern.reset(new mpi_future_internal<T>(const_cast<T&>(local_value)));


    impl::_check_mpi_result(
                MPI_Isend(_to_void_pointer(flat_aspect.flat()), static_cast<int>(flat_aspect.get_flat_size()),
                   _mpi_datatype_mapper(local_value),
                  dest_node, tag, _comm, &(fut.get_request())),
                  EIO,
                 "Error during MPI_send() "
    );

    return fut;
}


template <typename T>
inline mpi_future<T*> mpi_comm::send_async(const T * value, std::size_t n_value ,
                 int dest_node, int tag){

    using namespace impl;

    mpi_future<T*> fut;
    fut._intern.reset(new mpi_future_internal<T*>(value));


    impl::_check_mpi_result(
                MPI_Isend(_to_void_pointer(value), static_cast<int>(n_value),
                   _mpi_datatype_mapper(*value),
                  dest_node, tag, _comm, &(fut.get_request())),
                  EIO,
                 "Error during MPI_send() "
    );

    return fut;
}





inline mpi_comm::message_handle mpi_comm::probe(int src_node, int tag){
    mpi_comm::message_handle handle;

    if( MPI_Mprobe(src_node, tag, _comm, &(handle._msg), &(handle._status)) != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_Mprobe() ");
    }
    return handle;
}


inline mpi_comm::message_handle mpi_comm::probe(int src_node, int tag, std::size_t us_time){
    mpi_comm::message_handle handle;
    int flag =0;

    do{

        if( MPI_Improbe(src_node, tag, _comm, &flag, &(handle._msg), &(handle._status)) != MPI_SUCCESS){
            throw mpi_exception(EIO, "Error during MPI_Mprobe() ");
        }
        if(flag || (us_time == 0)){
            break;
        }

        usleep(1);
        us_time -=1;
    }while(1);

    return handle;
}





template <typename T>
inline void mpi_comm::recv(int src_node, int tag, T & value){
    impl::_mpi_flaterize<T> flat_aspect(value);

    if(flat_aspect.is_static_size()){
        recv(src_node, tag, flat_aspect.flat(), flat_aspect.get_flat_size());
    }else{
        throw mpi_exception(ENOSYS, "Operation not supported now");
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



template <typename T>
inline mpi_future<T> mpi_comm::recv_async(int src_node, int tag, T & value){
    impl::_mpi_flaterize<T> flat_aspect(value);

    if(flat_aspect.is_static_size()){
        mpi_future<T> fut;
        fut._intern.reset(new mpi_future_internal<T>(value));

        impl::_check_mpi_result(MPI_Irecv(static_cast<void*>(&value), 1,
                                          impl::_mpi_datatype_mapper(value), src_node, tag, _comm, &(fut._intern->get_ref_req())),
                                EIO,
                                "Error during MPI_Irecv() "
        );
        return fut;

    }else{
        throw mpi_exception(ENOSYS, "Operation not supported now, please use probe() for variable size async recv()");
    }
}

template <typename T>
inline mpi_future<T> mpi_comm::recv_async(const mpi_comm::message_handle &handle, T & value){
    mpi_future<T> fut;

    impl::_mpi_flaterize<T> flat_aspect(value);

    flat_aspect.resize(handle.count<typename impl::_mpi_flaterize<T>::base_type>());

    impl::_check_mpi_result(
                MPI_Imrecv(static_cast<void*>(flat_aspect.flat()), handle.count<typename impl::_mpi_flaterize<T>::base_type>(),
                    impl::_mpi_datatype_mapper(*(flat_aspect.flat())),
                    const_cast<MPI_Message*>(&(handle._msg)), &(fut._req)),
                EIO,
                "Error during MPI_Imrecv() "
    );


    fut.set_validity(true);
    fut._v = &value;
    return fut;

}


template <typename T>
inline mpi_future<T*> mpi_comm::recv_async(int src_node, int tag, T* value, std::size_t n_value){
    mpi_future<T*> fut;
    fut._intern.reset(new mpi_future_internal<T*>(value));

    impl::_check_mpi_result(MPI_Irecv(static_cast<void*>(value), n_value,
                                      impl::_mpi_datatype_mapper(*value), src_node, tag, _comm, &(fut._intern->get_ref_req())),
                            EIO,
                            "Error during MPI_Irecv() "
    );

    return fut;
}




inline void mpi_comm::barrier(){
    if(MPI_Barrier(_comm) != MPI_SUCCESS){
        throw mpi_exception(EIO, "Error during MPI_barrier() ");
    }
}


} // mpi



#endif



