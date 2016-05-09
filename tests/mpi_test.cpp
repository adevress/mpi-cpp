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

    const int sender_rank = ((runtime.is_master() ==false)?(runtime.rank()-1):(runtime.size()-1));
    BOOST_CHECK_EQUAL(handle.get_tag(), 42);
    BOOST_CHECK_EQUAL(handle.get_rank(),  sender_rank);


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
        for(int i=0; i < max_n; ++i){
            vec.push_back(T(i));
        }
    }

    runtime.barrier();

    runtime.broadcast(vec, 0);


    BOOST_CHECK_EQUAL(vec.size(), max_n);

    std::cout << "vec_bcast_size:" << max_n << "\n";

    for(int i=0; i < max_n; ++i){
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

