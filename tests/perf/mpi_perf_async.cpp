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


#include <mpi-cpp/mpi.hpp>
#include <boost/chrono.hpp>
#include <stdexcept>



using namespace mpi;

void mpi_async_multiple_wait_any(std::size_t n_send)
{


    mpi_comm runtime;

    runtime.barrier();


    if(runtime.size()  <= 1){
        throw std::runtime_error("Impossible to execute perf test, mpi size need to be > 1 ");
        return;
    }


    std::vector<std::size_t> values_send(n_send);

    std::vector< mpi_future<size_t> > futures;

    std::vector<std::size_t> values_recv(n_send, 0);

    const int dest_node = (runtime.rank()+1 == runtime.size() )?(0):(runtime.rank()+1);

    for(std::size_t i =0; i < n_send; ++i){
        values_send[i] = i;
        futures.push_back(runtime.send_async(values_send[i], dest_node, 45) );

        futures.push_back(runtime.recv_async(any_source, 45, values_recv[i]) );
    }

    std::cout << "started all async, begin waiting " << std::endl;

    std::vector< mpi_future<size_t> >  completed_futures;

    while(futures.size() > 0){

       typedef std::vector< mpi_future<size_t> >::iterator future_it;

       std::vector<future_it> my_futures
               = mpi_future<size_t>::wait_some(futures);

       for(std::vector<future_it>::iterator it = my_futures.begin(); it < my_futures.end(); ++it){
            completed_futures.push_back(**it);
       }

       futures = mpi_future<size_t>::filter_invalid(futures);
      // std::cout << " " << completed_futures.size() << " futures completed" << std::endl;

    }


    std::size_t sum_recv= std::accumulate(values_recv.begin(), values_recv.end(), 0);
    std::size_t sum_send= std::accumulate(values_send.begin(), values_send.end(), 0);
    if(sum_recv != sum_send){
        throw std::runtime_error("Invalid recv != send check");
    }
    runtime.barrier();
}


int main(int argc, char** argv){

    mpi_scope_env _env(&argc, &argv);

    mpi_comm runtime;


    boost::chrono::system_clock::time_point t1 = boost::chrono::system_clock::now();

    std::size_t n_send= 20000;
    mpi_async_multiple_wait_any(n_send);


    boost::chrono::system_clock::time_point t2 = boost::chrono::system_clock::now();

    boost::chrono::duration<double> sec = t2 - t1;
    std::cout << "n_request: " << n_send << std::endl;
    std::cout << "time: " << sec.count() << " seconds" << std::endl;
    std::cout << "requests/s/node: " << (double(n_send))/sec.count()  << std::endl;
    std::cout << "global requests/s: " << (double(n_send))/sec.count()*runtime.size()  << std::endl;
}

