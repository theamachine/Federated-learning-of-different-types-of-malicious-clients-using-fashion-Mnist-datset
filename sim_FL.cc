#include "ns3/core-module.h"
#include "ns3/ns3-ai-module.h"
#include "ns3/log.h"
#include <iomanip>

using namespace std;
using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ELG_5142_NS3AI_Part2");

struct Env
{
    int client_num;
    int max_data;
    int data_id; 
    float validation_split;
}Packed;

struct Act
{
    float accuracy;
    int epoch;
}Packed;


class FL : public Ns3AIRL<Env, Act>
{
public:
    FL(uint16_t id);
    float Func(int client_num, int max_data, int data_id, float validation_split);
};

FL::FL(uint16_t id) : Ns3AIRL<Env, Act>(id) {
    SetCond(2, 0);      ///< Set the operation lock (even for ns-3 and odd for python).
}

float FL::Func(int client_num, int max_data, int data_id, float validation_split)
{
    auto env = EnvSetterCond();     ///< Acquire the Env memory for writing
    env->client_num = client_num;
    env->max_data = max_data;
    env->data_id = data_id;
    env->validation_split = validation_split;

    SetCompleted();                 ///< Release the memory and update conters

    auto act = ActionGetterCond();  ///< Acquire the Act memory for reading
    float global_accuracy = act->accuracy;
    int epoch = act->epoch;
    GetCompleted();                 ///< Release the memory, roll back memory version and update conters

    cout<<"//////////////////////////////////////////"<<endl;
    cout<<"C++ RECEIVES DATA"<<endl;
    cout<<"epoch number is:"<<epoch<<"\t";
    
    return global_accuracy;
}

int main(int argc, char *argv[])
{
    int memblock_key = 2333;        ///< memory block key, need to keep the same in the python script
    int client_num = 1;
    int max_data = 2;
    float validation_split = 0.1;

    // 1:mnist 2:cifar10
    int  data_id = 1;

    CommandLine cmd;
    cmd.AddValue ("client_num","number of clients",client_num);
    cmd.AddValue ("max_data","number of data fr use",max_data);
    cmd.AddValue ("validation_split","validation_split",validation_split);
    cmd.AddValue ("data_id","Data Id",data_id);
    cmd.Parse (argc, argv);

    FL fl(memblock_key);
    float global_accuracy = fl.Func(client_num, max_data, data_id, validation_split);
    fl.SetFinish();

    std::cout << std::setprecision(3);
    cout<<"Global accuracy is: "<<global_accuracy<<endl;
    cout<<"//////////////////////////////////////////"<<endl;
    return 0;
}
