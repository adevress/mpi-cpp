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

#define BOOST_TEST_MODULE mpiTests
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <mpi-cpp/mpi.hpp>

int argc = boost::unit_test::framework::master_test_suite().argc;
char ** argv = boost::unit_test::framework::master_test_suite().argv;

using namespace mpi;


struct MpiFixture{
    MpiFixture():  _env(&argc, &argv){
        _env.enable_exception_report();
    }

    ~MpiFixture(){

    }

    mpi_scope_env _env;
};



BOOST_GLOBAL_FIXTURE( MpiFixture);


template<typename T>
T conv_or_max_integral(const int val){
    if( std::numeric_limits<T>::digits >= std::numeric_limits<int>::digits ){
        return T(val);
    }
    if( val > int(std::numeric_limits<T>::max()) ){
        return std::numeric_limits<T>::max();
    }
    return T(val);
}


BOOST_AUTO_TEST_CASE( mpiTests )
{

    mpi_comm runtime;
    const int rank = runtime.rank();
    const int size = runtime.size();

    BOOST_CHECK(rank >=0);
    BOOST_CHECK(size > 0);
    BOOST_CHECK(rank < size);

    std::cout << " rank:" << rank << " size:" << size;

    runtime.barrier();

}


BOOST_AUTO_TEST_CASE( mpiMax)
{

    mpi_comm runtime;
    const int rank = runtime.rank();
    const int size = runtime.size();

    int proc_number = (rank+1)*10;
    int max_proc_number = 0;

    max_proc_number = runtime.all_max(proc_number);

    BOOST_CHECK(max_proc_number == (size)*10);




}


BOOST_AUTO_TEST_CASE( mpiSum)
{

    mpi_comm runtime;

    const int rank = runtime.rank();
    const int size = runtime.size();

    int proc_number = (rank+1)*10;
    int sum_proc_number=0;
    for(int i =1; i < runtime.size()+1; ++i)
        sum_proc_number+=i*10;

    int sum_all = runtime.all_sum(proc_number);


    size_t sum_size = runtime.all_sum(static_cast<size_t>(size));


    BOOST_CHECK(sum_proc_number == sum_all);
    BOOST_CHECK(sum_size == static_cast<size_t>(size)* static_cast<size_t>(size));


}


BOOST_AUTO_TEST_CASE( mpi_all_gather_int)
{

    mpi_comm runtime;

    int rank = runtime.rank();

    std::vector<int> vals;
    runtime.all_gather(rank, vals);

    BOOST_CHECK(vals.size() >=1 );
    BOOST_CHECK(int(vals.size()) == runtime.size());

    for(int i = 0 ; i < int(vals.size()); ++i){
        BOOST_CHECK_EQUAL(vals[i], i);
    }

}


typedef boost::mpl::list<char, unsigned char,
                        int, unsigned int,
                        long, unsigned long,
                        long long, unsigned long long,
                        float, double,
                        long double> test_types;

BOOST_AUTO_TEST_CASE_TEMPLATE( mpi_all_gather_number, T, test_types )
{
    mpi_comm runtime;

    int rank = runtime.rank();

    std::vector<T> vals;
    runtime.all_gather<T,T>( conv_or_max_integral<T>(rank), vals );

    BOOST_CHECK(vals.size() >=1 );
    BOOST_CHECK(int(vals.size()) == runtime.size());

    for(int i = 0 ; i < int(vals.size()); ++i){
        BOOST_CHECK_EQUAL(vals[i], conv_or_max_integral<T>(i) );
    }
}


BOOST_AUTO_TEST_CASE( mpi_all_gatherv_int)
{

    mpi_comm runtime;

    std::size_t size_result=0;
    int rank = runtime.rank();

    std::vector<int> intput_val(rank+1, rank+1);

    std::vector<int> vals;
    runtime.all_gather(intput_val, vals);

    for(int i=0; i < runtime.size(); ++i){
        size_result += i+1;
    }

    BOOST_CHECK_EQUAL(vals.size(), size_result);


    int counter= 0, value= 1;
    for(size_t i = 0 ; i < vals.size(); ++i){
        if(counter >= value){
            counter= 0;
            value+= 1;
        }
        counter++;
        BOOST_CHECK_EQUAL(vals[i], value);
    }

}




BOOST_AUTO_TEST_CASE( mpi_send_recv_ring_int )
{
    mpi_comm runtime;

    if(runtime.size() ==1){
        std::cout << "Only one single node mpi_send_recv_ring can not be executed\n";
        return;
    }

    int rank = runtime.rank();
    int next_rank = ((rank +1 == runtime.size())?0:rank+1);

    if(runtime.is_master())
        runtime.send(0, next_rank, 256);

    int v;
    runtime.recv(any_source, any_tag, v);

    v = v+1;

    std::cout << "recv_val:" << v << "\n";

    if(runtime.is_master() == false){
        runtime.send(v, next_rank, 256);
    }


    if(runtime.is_master()){
        BOOST_CHECK_EQUAL(v, runtime.size() );
    } else{
        BOOST_CHECK_EQUAL(v, rank );
    }

}


BOOST_AUTO_TEST_CASE_TEMPLATE( mpi_send_recv_ring, T, test_types )
{
    mpi_comm runtime;

    if(runtime.size() ==1){
        std::cout << "Only one single node mpi_send_recv_ring can not be executed\n";
        return;
    }

    int rank = runtime.rank();
    int next_rank = ((rank +1 == runtime.size())?0:rank+1);

    if(runtime.is_master())
        runtime.send(T(0), next_rank, 256);

    T v;
    runtime.recv(any_source, any_tag, v);

    v = (( std::numeric_limits<T>::max() == (v) )?v:v+1);

    std::cout << "recv_val:" << v << "\n";

    if(runtime.is_master() == false){
        runtime.send(v, next_rank, 256);
    }



    if(runtime.is_master()){
        BOOST_CHECK_EQUAL(v, conv_or_max_integral<T>(runtime.size()) );
    } else{
        BOOST_CHECK_EQUAL(v, conv_or_max_integral<T>(rank) );
    }

}

BOOST_AUTO_TEST_CASE( mpi_send_recv_ring_string_probe)
{
    mpi_comm runtime;

    if(runtime.size() ==1){
        std::cout << "Only one single node mpi_send_recv_ring can not be executed\n";
        return;
    }

    int rank = runtime.rank();
    int next_rank = ((rank +1 == runtime.size())?0:rank+1);


    std::string buffer;

    if(runtime.is_master()){

        buffer.push_back('a');
        runtime.send(buffer, next_rank, 42);
    }

    mpi_comm::message_handle handle = runtime.probe(any_source, any_tag);
    BOOST_CHECK(handle.is_valid());

    const int sender_rank = ((runtime.is_master() ==false)?(runtime.rank()-1):(runtime.size()-1));
    BOOST_CHECK_EQUAL(handle.tag(), 42);
    BOOST_CHECK_EQUAL(handle.rank(),  sender_rank);


    runtime.recv(handle, buffer);

    std::cout << "recv_str_val:" << buffer << "\n";

    if(runtime.is_master() == false){
        std::string buffer_send(buffer);
        int pos = runtime.rank()%26;

        for(int i =-1; i < runtime.rank();++i){
            buffer_send.push_back('a' + pos);
        }
        runtime.send(buffer_send, next_rank, 42);
    }



    const int actors = ((runtime.is_master())?(runtime.size()):(runtime.rank()));


    BOOST_CHECK_EQUAL(buffer.size(), (actors*(actors+1))/2);

}





BOOST_AUTO_TEST_CASE_TEMPLATE( mpi_bcast_sync_100, T, test_types )
{
    mpi_comm runtime;


    runtime.barrier();

    for(int i =0; i <  100; ++i){

        T num = T(i);
        T buffer;
        if(runtime.is_master()){
            buffer = num;
        }
        runtime.broadcast(&buffer, 1, 0);

        std::cout << "bcast:" << buffer << "\n";

        BOOST_CHECK_EQUAL(buffer, num);

        runtime.barrier();
    }

}


BOOST_AUTO_TEST_CASE_TEMPLATE( mpi_bcast_vec, T, test_types )
{
    mpi_comm runtime;
    std::size_t max_n = 125;

    std::vector<T> vec;

    if(runtime.is_master()){
        for(std::size_t i=0; i < max_n; ++i){
            vec.push_back(T(i));
        }
    }

    runtime.barrier();

    runtime.broadcast(vec, 0);


    BOOST_CHECK_EQUAL(vec.size(), max_n);

    std::cout << "vec_bcast_size:" << max_n << "\n";

    for(std::size_t i=0; i < max_n; ++i){
        BOOST_CHECK_EQUAL(vec[i], T(i));
    }


}


BOOST_AUTO_TEST_CASE( broadcast_str )
{
    mpi_comm runtime;
    const std::string str_def = "Hello world! 你好";

    runtime.barrier();


    std::string value;

    if(runtime.is_master()){
        value = str_def;
    }

    runtime.broadcast(value, 0);



   BOOST_CHECK_EQUAL(value.size(), str_def.size());
   BOOST_CHECK_EQUAL(value, str_def);

   std::cout << "str_bcast:" << value << "\n";

}



BOOST_AUTO_TEST_CASE( mpi_im_probe_test )
{
    mpi_comm runtime;
    const int tag = 56;
    const std::string str_def = "Hello world! 你好";

    if(runtime.size() ==1){
        std::cout << "Only one single node, mpi_im_probe_test can not be executed\n";
        return;
    }

    runtime.barrier();

    if(runtime.is_master()){

        // check non-blocking version
        mpi_comm::message_handle handle =
                   runtime.probe(1, tag, 0);
        BOOST_CHECK(handle.is_valid() == false);

        // retry with positive value
        handle = runtime.probe(1, tag, 2000);
        BOOST_CHECK(handle.is_valid() == false);
    }

    runtime.barrier();

    if(runtime.rank() == 1){
        runtime.send(str_def, 0, tag);
    }

    if(runtime.is_master()){
        std::string res;
	mpi_comm::message_handle handle;
        BOOST_CHECK(handle.is_valid() == false);


	do{
	        handle =
	                   runtime.probe(1, tag, 5000);
	} while(handle.is_valid() == false);

        BOOST_CHECK(handle.is_valid() == true);

        runtime.recv(handle, res);
        BOOST_CHECK_EQUAL(res, str_def);
        std::cout << "probe_recv:" << res << "\n";
    }


    runtime.barrier();

}



BOOST_AUTO_TEST_CASE( mpi_simple_self_async )
{
    mpi_comm runtime;
    std::size_t value = 42, recv_value=0;

    mpi_comm::mpi_future<std::size_t> invalid_future;
    // any not initialized future should throw if accessed
    // and be invalid
    BOOST_CHECK_EQUAL(invalid_future.valid(), false);
    BOOST_CHECK_THROW({
                        invalid_future.wait();

                      }, mpi_invalid_future);


    mpi_comm::mpi_future<std::size_t> fut_recv
            = runtime.recv_async(any_source, any_tag, recv_value);

    mpi_comm::mpi_future<std::size_t> fut_send
            = runtime.send_async(value, runtime.rank(), 2);


    fut_send.wait();
    fut_recv.wait();

    BOOST_CHECK_EQUAL(fut_recv.valid(), true);
    BOOST_CHECK_EQUAL(fut_send.valid(), true);

    BOOST_CHECK_EQUAL(fut_recv.get(), value);

    BOOST_CHECK_EQUAL(recv_value, value);

    fut_send.get();

    std::cout << "recv_async_val " << recv_value << std::endl;

    // second get() on future should throw
    BOOST_CHECK_THROW({
                        const std::size_t val = fut_recv.get();
                        (void) val;

                      }, mpi_invalid_future);


}





BOOST_AUTO_TEST_CASE( mpi_non_blocking_self )
{
    mpi_comm runtime;
    std::size_t value = 42, recv_value=0;

    runtime.barrier();


    mpi_comm::mpi_future<std::size_t> fut_recv
            = runtime.recv_async(any_source, any_tag, recv_value);

    mpi_comm::mpi_future<std::size_t> fut_send
            = runtime.send_async(value, runtime.rank(), 2);


    // due continuous polling without timeout
    // should work if wait_for execute a MPI_Test even for 0
    while(fut_recv.wait_for(0) == false);
    fut_send.wait();

    BOOST_CHECK_EQUAL(fut_recv.get(), value);

    BOOST_CHECK_EQUAL(recv_value, value);

    std::cout << "recv_async_val " << recv_value << std::endl;

    // second get() on future should throw
    BOOST_CHECK_THROW({
                        const std::size_t val = fut_recv.get();
                        (void) val;

                      }, mpi_invalid_future);

    runtime.barrier();

}




BOOST_AUTO_TEST_CASE( mpi_future_lifetime_check)
{
    mpi_comm runtime;
    std::size_t value = 144, recv_value=0;

    runtime.barrier();

    mpi_comm::mpi_future<std::size_t> other_future;


   mpi_comm::mpi_future<std::size_t> fut_recv
            = runtime.recv_async(any_source, any_tag, recv_value);


   // check validity even before operation
   BOOST_CHECK_EQUAL(fut_recv.valid(), true);

   // and other invalidity
   BOOST_CHECK_EQUAL(other_future.valid(), false);

   // now we copy recv future to invalid future
   // invalid
    other_future = fut_recv;

    // roles are inverted now

    // check validity even before operation
    BOOST_CHECK_EQUAL(fut_recv.valid(), false);

    // and other invalidity
    BOOST_CHECK_EQUAL(other_future.valid(), true);

    // wait or on recv should not work

    // second get() on future should throw
    BOOST_CHECK_THROW({
                        const std::size_t val = fut_recv.get();
                        (void) val;

                      }, mpi_invalid_future);

    BOOST_CHECK_THROW({
                        fut_recv.wait();

                      }, mpi_invalid_future);



   mpi_comm::mpi_future<std::size_t>
           fut_send = runtime.send_async(value, runtime.rank(), 2);



    other_future.wait();
    fut_send.wait();

    BOOST_CHECK_EQUAL(other_future.valid(), true);

    BOOST_CHECK_EQUAL(other_future.get(), value);

    //

    std::vector< mpi_comm::mpi_future<std::size_t> > vec_future(10);

    for(std::vector<mpi_comm::mpi_future<std::size_t> >::iterator it = vec_future.begin();
        it < vec_future.end(); ++it){
        BOOST_CHECK_EQUAL(it->valid(), false);
    }

    vec_future.push_back(fut_send);

    // fut send has been copied, should now be invalid
    BOOST_CHECK_EQUAL(fut_send.valid(), false);

    // lets check the result of the copy
    BOOST_CHECK_EQUAL(vec_future.back().get(), value);

    runtime.barrier();

}



BOOST_AUTO_TEST_CASE( mpi_async_multiple_simpl)
{


    mpi_comm runtime;
    const std::size_t n_send = 200;


    runtime.barrier();

    std::vector<std::size_t> values_send(n_send);
    std::vector< mpi_comm::mpi_future<size_t> > send_futures(n_send);

    std::vector<std::size_t> values_recv(n_send, 0);
    std::vector< mpi_comm::mpi_future<size_t> > recv_futures(n_send);

    const int dest_node = (runtime.rank()+1 == runtime.size() )?(0):(runtime.rank()+1);

    for(std::size_t i =0; i < n_send; ++i){
        values_send[i] = i;
        send_futures[i] = runtime.send_async(values_send[i], dest_node, 44);
    }

    for(std::size_t i =0; i < n_send; ++i){
        recv_futures[i] = runtime.recv_async(any_source, 44, values_recv[i]);
    }

    for(std::size_t i = 0; i < n_send; ++i){
        send_futures[i].wait();
        recv_futures[i].wait();
    }


    std::size_t sum_recv= std::accumulate(values_recv.begin(), values_recv.end(), 0);
    std::size_t sum_send= std::accumulate(values_send.begin(), values_send.end(), 0);

    BOOST_CHECK_EQUAL(sum_recv, sum_send);

    std::cout << "sum_async_all:" << sum_recv << " " << sum_send << "\n";



    runtime.barrier();
}


BOOST_AUTO_TEST_CASE( mpi_async_multiple_wait_some)
{


    mpi_comm runtime;
    const std::size_t n_send = 400;


    runtime.barrier();

    std::vector<std::size_t> values_send(n_send);

    std::vector< mpi_comm::mpi_future<size_t> > futures;

    std::vector<std::size_t> values_recv(n_send, 0);

    const int dest_node = (runtime.rank()+1 == runtime.size() )?(0):(runtime.rank()+1);

    for(std::size_t i =0; i < n_send; ++i){
        values_send[i] = i;
        futures.push_back(runtime.send_async(values_send[i], dest_node, 43) );

        futures.push_back(runtime.recv_async(any_source, 43, values_recv[i]) );
    }


    std::vector< mpi_comm::mpi_future<size_t> >  completed_futures;

    while(futures.size() > 0){

       std::vector< mpi_comm::mpi_future<size_t> > triggered
               = mpi_comm::mpi_future<size_t>::wait_some(futures);

       std::cout << "multiple_wait_some:" << triggered.size() << "\n";

       completed_futures.insert(completed_futures.end(), triggered.begin(), triggered.end());

       futures = mpi_comm::mpi_future<size_t>::filter_invalid(futures);

       std::cout << "multiple_wait_some_remaining:" << futures.size() << "\n";
    }


    std::size_t sum_recv= std::accumulate(values_recv.begin(), values_recv.end(), 0);
    std::size_t sum_send= std::accumulate(values_send.begin(), values_send.end(), 0);

    BOOST_CHECK_EQUAL(sum_recv, sum_send);

    runtime.barrier();
}




BOOST_AUTO_TEST_CASE( mpi_async_multiple_wait_any)
{


    mpi_comm runtime;
    const std::size_t n_send = 200;


    runtime.barrier();

    std::vector<std::size_t> values_send(n_send);

    std::vector< mpi_comm::mpi_future<size_t> > futures;

    std::vector<std::size_t> values_recv(n_send, 0);

    const int dest_node = (runtime.rank()+1 == runtime.size() )?(0):(runtime.rank()+1);

    for(std::size_t i =0; i < n_send; ++i){
        values_send[i] = i;
        futures.push_back(runtime.send_async(values_send[i], dest_node, 45) );

        futures.push_back(runtime.recv_async(any_source, 45, values_recv[i]) );
    }


    std::vector< mpi_comm::mpi_future<size_t> >  completed_futures;

    while(futures.size() > 0){

       mpi_comm::mpi_future<size_t>  my_future
               = mpi_comm::mpi_future<size_t>::wait_any(futures);

       completed_futures.push_back(my_future);

       std::size_t pre_filter_size = futures.size();

       futures = mpi_comm::mpi_future<size_t>::filter_invalid(futures);

       BOOST_CHECK_EQUAL(futures.size(), pre_filter_size-1);

    }


    std::size_t sum_recv= std::accumulate(values_recv.begin(), values_recv.end(), 0);
    std::size_t sum_send= std::accumulate(values_send.begin(), values_send.end(), 0);

    BOOST_CHECK_EQUAL(sum_recv, sum_send);

    runtime.barrier();
}




