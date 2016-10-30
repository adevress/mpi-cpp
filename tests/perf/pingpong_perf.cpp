#include <mpi-cpp/mpi.hpp>


#include <boost/lexical_cast.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>


typedef std::vector<char> vector_elems;

struct exec_context{
    exec_context() :
        iterations(1000),
        n_parallel(1),
        elem_size(1),
        min_time(1000000),
        max_time(0),
        av_time(0) {}


    std::size_t iterations, n_parallel;
    std::size_t elem_size;
    double min_time, max_time, av_time;
};


template<typename Func>
void execute_iter(const std::string name, Func & fun){
    mpi::mpi_comm comm;

    std::size_t max_elem_size = 65536;

    comm.barrier();

    if(comm.rank() == 0) {
        std::cout << " " << name << "\n";

        std::cout << "\t\t#bytes\t\t#repetitions\t\tt[usec]\t\tmint[usec]\t\tmaxt[usec]\t\tMbytes/sec\n";
    }



    std::size_t elem_size = 0;

    while(elem_size <= max_elem_size){
        exec_context context;
        context.elem_size = elem_size;
        fun(context);

        if(comm.rank() == 0) {
            std::cout << "\t\t" << context.elem_size
                  << "\t\t" << context.iterations
                  << "\t\t" << context.av_time
                  << "\t\t" << context.min_time
                  << "\t\t" << context.max_time
                  << "\t\t" << (context.elem_size) / ( context.av_time )
                  << "\n";
        }
        elem_size= (elem_size == 0 ) ? 1 : (elem_size << 1);
    }

    std::cout << std::endl;

    comm.barrier();

}

void synchronous_dynqmic_execute_ping_pong(exec_context & c){
    std::size_t total = 0;

    vector_elems elems, elems_res;
    mpi::mpi_comm comm;

    for(std::size_t i =0; i < c.elem_size; ++i){
        elems.push_back(char(i));
    }



    for(std::size_t i = 0; i < c.iterations; i++){
        auto start = std::chrono::system_clock::now();

        if(comm.rank() == 0) {

            comm.send(elems, 1, 42);

            auto handle = comm.probe(mpi::any_source, mpi::any_tag);
            comm.recv(handle, elems_res);
            total += elems_res.size();

        }else if(comm.rank() == 1){

            auto handle = comm.probe(mpi::any_source, mpi::any_tag);
            comm.recv(handle, elems_res);

            comm.send(elems_res, 0, 42);

            total += elems_res.size();
        }

       auto stop = std::chrono::system_clock::now();
       double time_us = double(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()) / 1000.0 ;
       c.min_time = std::min(c.min_time, time_us);
       c.max_time = std::max(c.max_time, time_us);
       c.av_time += time_us;
    }


    c.av_time /= c.iterations;

}


void synchronous_fixed_execute_ping_pong(exec_context & c){
    std::size_t total = 0;

    vector_elems elems, elems_res;
    mpi::mpi_comm comm;

    for(std::size_t i =0; i < c.elem_size; ++i){
        elems.push_back(char(i));
    }

    elems_res.resize(c.elem_size+1);


    comm.barrier();
    comm.barrier();

    for(std::size_t i = 0; i < c.iterations; i++){
        auto start = std::chrono::system_clock::now();

        if(comm.rank() == 0) {

            comm.send(elems.data(), c.elem_size, 1, 42);

            comm.recv(mpi::any_source, mpi::any_tag, elems_res.data(), c.elem_size);
            total += elems_res.size();

        }else if(comm.rank() == 1){

            comm.recv(mpi::any_source, mpi::any_tag, elems_res.data(), c.elem_size);

            comm.send(elems_res.data(), c.elem_size,  0, 42);

            total += elems_res.size();
        }

       auto stop = std::chrono::system_clock::now();
       double time_us = double(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()) / 1000.0 ;
       c.min_time = std::min(c.min_time, time_us);
       c.max_time = std::max(c.max_time, time_us);
       c.av_time += time_us;
    }


    c.av_time /= c.iterations;

    comm.barrier();
    comm.barrier();

}



void asynchronous_fixed_execute_ping_pong(exec_context & c){
    std::size_t total = 0;

    vector_elems elems, elems_res;
    mpi::mpi_comm comm;

    for(std::size_t i =0; i < c.elem_size; ++i){
        elems.push_back(char(i));
    }

    elems_res.resize(c.elem_size+1);


    comm.barrier();
    comm.barrier();

    for(std::size_t i = 0; i < c.iterations; i++){
        auto start = std::chrono::system_clock::now();

        if(comm.rank() == 0) {

            auto fut_send = comm.send_async(elems.data(), c.elem_size, 1, 42);

            auto fut_recv = comm.recv_async(mpi::any_source, mpi::any_tag, elems_res.data(), c.elem_size);
            total += elems_res.size();

            fut_send.wait();
            fut_recv.wait();

        }else if(comm.rank() == 1){

            auto fut_recv = comm.recv_async(mpi::any_source, mpi::any_tag, elems_res.data(), c.elem_size);

            fut_recv.wait();

            auto fut_send = comm.send_async(elems_res.data(), c.elem_size,  0, 42);

            total += elems_res.size();

            fut_send.wait();

        }

       auto stop = std::chrono::system_clock::now();
       double time_us = double(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()) / 1000.0 ;
       c.min_time = std::min(c.min_time, time_us);
       c.max_time = std::max(c.max_time, time_us);
       c.av_time += time_us;
    }


    c.av_time /= c.iterations;

    comm.barrier();
    comm.barrier();

}



void asynchronous_dynamic_execute_ping_pong(exec_context & c){
    std::size_t total = 0;

    vector_elems elems, elems_res;
    mpi::mpi_comm comm;

    for(std::size_t i =0; i < c.elem_size; ++i){
        elems.push_back(char(i));
    }

    elems_res.reserve(c.elem_size);

    comm.barrier();
    comm.barrier();

    for(std::size_t i = 0; i < c.iterations; i++){
        auto start = std::chrono::system_clock::now();

        if(comm.rank() == 0) {

            auto fut_send = comm.send_async(elems, 1, 42);

            auto msg_handle = comm.probe(mpi::any_source, mpi::any_tag);

            auto fut_recv = comm.recv_async<vector_elems>(msg_handle);
            total += elems_res.size();

            fut_send.wait();
            fut_recv.wait();

        }else if(comm.rank() == 1){

            auto msg_handle = comm.probe(mpi::any_source, mpi::any_tag);
            auto fut_recv = comm.recv_async<vector_elems>(msg_handle);

            fut_recv.wait();

            auto fut_send = comm.send_async(elems_res.data(), c.elem_size,  0, 42);

            total += elems_res.size();

            fut_send.wait();

        }

       auto stop = std::chrono::system_clock::now();
       double time_us = double(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()) / 1000.0 ;
       c.min_time = std::min(c.min_time, time_us);
       c.max_time = std::max(c.max_time, time_us);
       c.av_time += time_us;
    }


    c.av_time /= c.iterations;

    comm.barrier();
    comm.barrier();

}

int main(int argc, char** argv)
{

    mpi::mpi_scope_env mpi_env(&argc, &argv);


    execute_iter("Synchronous fixed size", synchronous_fixed_execute_ping_pong);
    execute_iter("Synchronous dynamic sized", synchronous_dynqmic_execute_ping_pong);
    execute_iter("Asynchronous fixed size", asynchronous_fixed_execute_ping_pong);
    execute_iter("Asynchronous dynamic size", asynchronous_fixed_execute_ping_pong);

}


