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
#ifndef MPI_EXCEPTION_HPP
#define MPI_EXCEPTION_HPP

#include <stdexcept>
#include <exception>
#include <cerrno>

///
/// \brief generic mpi exception class
///
class mpi_exception: public std::runtime_error {
public:
    inline mpi_exception(int code, const std::string & msg): std::runtime_error(msg), _code(code){}
    inline virtual ~mpi_exception() throw() {}

    inline int value() const { return _code; }

private:
    int _code;
};


///
/// \brief The mpi_invalid_future class
///
class mpi_invalid_future: public mpi_exception {
public:
    inline mpi_invalid_future() :
        mpi_exception(EINVAL, "wait() or get() executed on invalid mpi_future object"){}

private:
};


#endif // MPI_EXCEPTION_HPP
