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
#ifndef HADOKEN_MPI_COMM_HPP
#define HADOKEN_MPI_COMM_HPP

#include <mpi.h>

#include <boost/noncopyable.hpp>
#include "mpi_exception.hpp"


namespace mpi {


/**
 * @brief mpi environment scoper
 *
 *  initialize MPI when constructed, finalize MPI when destroyed
 */
class mpi_scope_env : private boost::noncopyable{
public:
    inline mpi_scope_env(int* argc, char*** argv);

    inline virtual ~mpi_scope_env();

private:
    int initialized;
};

/**
 * @brief mpi communicator
 *
 * bridge to all the MPI functions
 */
class mpi_comm : private boost::noncopyable
{
public:

    /**
     * @brief mpi_comm
     *
     * construct an MPI communication channel
     *
     * MPI operation
     */

    inline mpi_comm();
    inline virtual ~mpi_comm();

    /**
     * @brief rank
     * @return mpi rank of the process
     */
    inline int rank() const{
        return _rank;
    }

    /**
     * @brief size
     * @return size of the mpi communication domain
     */
    inline int size() const{
        return _size;
    }


    /**
     * @brief isMaster
     * @return true if current node is root ( 0 )
     */
    inline bool is_master() const{
        return _rank ==0;
    }

    /// @brief synchronization barrier
    ///
    /// blocking until all the nodes reach it
    inline void barrier();


    /// send local_value to node dest
    /// @param local_value value to send
    /// @param dest node id of the reciver
    /// @param tag identity tag
    template <typename T>
    inline void send(const T & local_value, int dest_node, int tag);

    template <typename T>
    inline void send(const T * value, std::size_t n_value ,
                     int dest_node, int tag);

    /// received data from an other node
    /// @param src node id of the sender
    /// @param tag identity tag
    /// @return value received
    template <typename T>
    inline void recv(int src_node, int tag, T & value);

    template <typename T>
    inline void recv(int src_node, int tag, T* value, std::size_t n_value);


    /// return gathered information from all nodes
    /// to all nodes
    ///
    /// @return vector of all the gathered values
    ///
    /// collective operation
    template <typename T, typename Y>
    inline void all_gather(const T & input, std::vector<Y> & output);

    template <typename T, typename Y>
    inline void all_gather(const T* ivalues, std::size_t nvalues, Y* ovalues);

    template <typename T, typename Y>
    inline void all_gather(const T* ivalues, std::size_t nvalues,
                                     Y* ovalues, const std::size_t * ovalues_per_node);


    /// @brief MPI broadcast
    ///
    ///  @param value to broadcast
    ///  @param root node
    ///
    ///  Broadcast a value from node root to all the nodes of
    ///  the communication handle
    ///
    ///
    template <typename T>
    inline void broadcast(T & values, int root);

    template <typename T>
    inline void broadcast(T* buffer, std::size_t nvalues, int root);



    /// @brief return the highest element of the nodes
    /// @param local_value value to compare
    /// @return highest element between all nodes
    ///
    /// collective operation
    template <typename T>
    inline T all_max(const T & elems);

    template <typename T>
    inline void all_max(const T* ivalue, std::size_t n_value, T* ovalue);


    /// @brief return the sum of all the elements of the nodes
    /// @param local_value value to sum
    /// @return sum of all the elements
    ///
    /// collective operation
    template <typename T>
    inline T all_sum(const T & local_value);

    template <typename T>
    inline void all_sum(const T* ivalue, std::size_t n_value, T* ovalue);

private:
    int _rank;
    int _size;
    MPI_Comm _comm;

};



/// various const definitions
const int any_tag = MPI_ANY_TAG;
const int any_source = MPI_ANY_SOURCE;

}


#include "impl/mpi.tcc"

#endif
