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
#ifndef MPI_TYPE_MAPPER_HPP
#define MPI_TYPE_MAPPER_HPP

#include <string>
#include <mpi.h>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_arithmetic.hpp>



namespace mpi{

namespace impl{


inline MPI_Datatype _mpi_datatype_mapper(char value){
    (void) value;
    return MPI_CHAR;
}

inline MPI_Datatype _mpi_datatype_mapper(unsigned char value){
    (void) value;
    return MPI_BYTE;
}


inline MPI_Datatype _mpi_datatype_mapper(int value){
    (void) value;
    return MPI_INT;
}

inline MPI_Datatype _mpi_datatype_mapper(unsigned int value){
    (void) value;
    return MPI_UNSIGNED;
}


inline MPI_Datatype _mpi_datatype_mapper(long value){
    (void) value;
    return MPI_LONG;
}

inline MPI_Datatype _mpi_datatype_mapper(unsigned long value){
    (void) value;
    return MPI_UNSIGNED_LONG;
}

inline MPI_Datatype _mpi_datatype_mapper(long long value){
    (void) value;
    return MPI_LONG_LONG;
}

inline MPI_Datatype _mpi_datatype_mapper(unsigned long long value){
    (void) value;
    return MPI_UNSIGNED_LONG_LONG;
}

inline MPI_Datatype _mpi_datatype_mapper(float value){
    (void) value;
    return MPI_FLOAT;
}

inline MPI_Datatype _mpi_datatype_mapper(double value){
    (void) value;
    return MPI_DOUBLE;
}

inline MPI_Datatype _mpi_datatype_mapper(long double value){
    (void) value;
    return MPI_LONG_DOUBLE;
}

template<typename T>
inline MPI_Datatype _mpi_datatype_mapper(const std::vector<T> & value){
    (void) value;
    return _mpi_datatype_mapper(T());
}


template<typename T>
inline MPI_Datatype _mpi_datatype_mapper(const std::string & value){
    (void) value;
    return MPI_CHAR;
}


//
// flaterizer
//

// dummy
template <class T, class Enable = void>
class _mpi_flaterize { };


// numerical types, single value
template <class T>
class _mpi_flaterize<T, typename boost::enable_if< boost::is_arithmetic<T> >::type>{
public:
    typedef T base_type;

    inline _mpi_flaterize(const T & integral_ref): _ref(integral_ref) {}

    inline base_type * flat(){
        return const_cast<T*>(&_ref);
    }

    inline std::size_t get_flat_size(){
        return 1;
    }

    inline bool is_static_size() const{
        return true;
    }


    void resize(std::size_t n_values){
        (void) n_values;
    }


private:
    const T & _ref;

};


//
// any vector of base type
// define vector as variable length
//
template <class T>
class _mpi_flaterize<T, typename boost::enable_if< boost::is_arithmetic<typename T::value_type> >::type>{
public:
    typedef typename T::value_type base_type;

    inline _mpi_flaterize(T & vec_values): _ref_vec(vec_values) {}

    inline base_type * flat(){
        return const_cast<base_type*>(&(_ref_vec[0]));
    }

    inline std::size_t get_flat_size(){
        return _ref_vec.size();
    }

    inline bool is_static_size() const{
        return false;
    }


    void resize(std::size_t n_values){
        _ref_vec.resize(n_values);
    }


private:
    std::vector<base_type> & _ref_vec;

};

//
// string mapper
//
template <>
class _mpi_flaterize<std::string>{
public:
    typedef char base_type;

    inline _mpi_flaterize(std::string & str): _ref_str(str) {}

    inline base_type * flat(){
        return const_cast<base_type*>(_ref_str.c_str());
    }

    inline std::size_t get_flat_size(){
        return _ref_str.size();
    }

    inline bool is_static_size() const{
        return false;
    }


    void resize(std::size_t n_values){
        _ref_str.resize(n_values);
    }


private:
    std::string & _ref_str;

};


} // impl

} // mpi


#endif // MPI_TYPE_MAPPER_HPP
