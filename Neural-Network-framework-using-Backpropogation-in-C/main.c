#include "backprop.h"
#include "layer.h"
#include "neuron.h"
#define TRAININGRPUND 20000     //此為電腦train的次數，我用define在上面比較好直接改數字


layer *lay = NULL;                       //layer 是自己定義的型別
int num_layers;
int *num_neurons;
float alpha;
float *cost;
float full_cost;
float **input;
float **desired_outputs;
int num_training_ex;
int n=1;

//以下的宣告是為了算mean square error，所以開一個excel檔    
float MSE;
float temp_MSE;
void compute_MSE(int i);
FILE *exfptr;           //excel file 的pionter


int main(void)
{
    int i;

    if((exfptr = fopen("MSE_compute_sheet.csv", "w")) == NULL){            //如果創建csv檔出錯，顯示File is dead
        printf("File is dead\n");
    }else{

        srand(time(0));

        printf("Enter the number of Layers in Neural Network:\n");
        scanf("%d",&num_layers);

        num_neurons = (int*) malloc(num_layers * sizeof(int));          //用 malloc 開儲存空間
        memset(num_neurons,0,num_layers *sizeof(int));                  //memset 來初始化

        // Get number of neurons per layer
        for(i=0;i<num_layers;i++)
        {
            printf("Enter number of neurons in layer[%d]: \n",i+1);
            scanf("%d",&num_neurons[i]);
        }

        printf("\n");

        // Initialize the neural network module
        if(init()!= SUCCESS_INIT)
        {
            printf("Error in Initialization...\n");
            exit(0);
        }

        printf("Enter the learning rate (Usually 0.15): \n");
        scanf("%f",&alpha);
        printf("\n");

        printf("Enter the number of training examples: \n");
        scanf("%d",&num_training_ex);
        printf("\n");

        input = (float**) malloc(num_training_ex * sizeof(float*));
        for(i=0;i<num_training_ex;i++)
        {
            input[i] = (float*)malloc(num_neurons[0] * sizeof(float));
        }

        desired_outputs = (float**) malloc(num_training_ex* sizeof(float*));
        for(i=0;i<num_training_ex;i++)
        {
            desired_outputs[i] = (float*)malloc(num_neurons[num_layers-1] * sizeof(float));
        }

        cost = (float *) malloc(num_neurons[num_layers-1] * sizeof(float));
        memset(cost,0,num_neurons[num_layers-1]*sizeof(float));

        // Get Training Examples
        get_inputs();

        // Get Output Labels
        get_desired_outputs();

        train_neural_net();
        test_nn();

        if(dinit()!= SUCCESS_DINIT)
        {
            printf("Error in Dinitialization...\n");
        }
    }

    fclose(exfptr);

    return 0;
}


int init()
{
    if(create_architecture() != SUCCESS_CREATE_ARCHITECTURE)
    {
        printf("Error in creating architecture...\n");
        return ERR_INIT;
    }

    printf("Neural Network Created Successfully...\n\n");
    return SUCCESS_INIT;
}

//Get Inputs
void  get_inputs()
{
    int i,j;

        for(i=0;i<num_training_ex;i++)
        {
            printf("Enter the Inputs for training example[%d]:\n",i);

            for(j=0;j<num_neurons[0];j++)
            {
                scanf("%f",&input[i][j]);
                
            }
            printf("\n");
        }
}

//Get Labels
void get_desired_outputs()
{
    int i,j;
    
    for(i=0;i<num_training_ex;i++)
    {
        for(j=0;j<num_neurons[num_layers-1];j++)
        {
            printf("Enter the Desired Outputs (Labels) for training example[%d]: \n",i);
            scanf("%f",&desired_outputs[i][j]);
            printf("\n");
        }
    }
}

// Feed inputs to input layer
void feed_input(int i)
{
    int j;

    for(j=0;j<num_neurons[0];j++)
    {
        lay[0].neu[j].actv = input[i][j];
        printf("Input: %f\n",lay[0].neu[j].actv);
    }
}

// Create Neural Network Architecture
int create_architecture()
{
    int i=0,j=0;
    lay = (layer*) malloc(num_layers * sizeof(layer));          //layer因為之前定義過了所以是型別喔

    for(i=0;i<num_layers;i++)
    {
        lay[i] = create_layer(num_neurons[i]);      
        lay[i].num_neu = num_neurons[i];
        printf("Created Layer: %d\n", i+1);
        printf("Number of Neurons in Layer %d: %d\n", i+1,lay[i].num_neu);

        for(j=0;j<num_neurons[i];j++)
        {
            if(i < (num_layers-1)) 
            {
                lay[i].neu[j] = create_neuron(num_neurons[i+1]);
            }

            printf("Neuron %d in Layer %d created\n",j+1,i+1);  
        }
        printf("\n");
    }

    printf("\n");

    // Initialize the weights
    if(initialize_weights() != SUCCESS_INIT_WEIGHTS)
    {
        printf("Error Initilizing weights...\n");
        return ERR_CREATE_ARCHITECTURE;
    }

    return SUCCESS_CREATE_ARCHITECTURE;
}

int initialize_weights(void)
{
    int i,j,k;

    if(lay == NULL)
    {
        printf("No layers in Neural Network...\n");
        return ERR_INIT_WEIGHTS;
    }

    printf("Initializing weights...\n");

    for(i=0;i<num_layers-1;i++)
    {
        
        for(j=0;j<num_neurons[i];j++)
        {
            for(k=0;k<num_neurons[i+1];k++)
            {
                // Initialize Output Weights for each neuron
                lay[i].neu[j].out_weights[k] = ((double)rand())/((double)RAND_MAX);
                printf("%d:w[%d][%d]: %f\n",k,i,j, lay[i].neu[j].out_weights[k]);
                lay[i].neu[j].dw[k] = 0.0;
            }

            if(i>0) 
            {
                lay[i].neu[j].bias = ((double)rand())/((double)RAND_MAX);
            }
        }
    }   
    printf("\n");
    
    for (j=0; j<num_neurons[num_layers-1]; j++)
    {
        lay[num_layers-1].neu[j].bias = ((double)rand())/((double)RAND_MAX);
    }

    return SUCCESS_INIT_WEIGHTS;
}

// Train Neural Network
void train_neural_net(void)
{
    int i;
    int it=0;
    fprintf(exfptr,"Num,MSE\n");

    // Gradient Descent
    for(it=0;it<TRAININGRPUND;it++)     //train 20000 次      為了觀察學習過程有時我調成200或50次
    {
        for(i=0;i<num_training_ex;i++)
        {
            feed_input(i);
            forward_prop();
            compute_cost(i);
            back_prop(i);
            update_weights();
        }

        compute_MSE(it);            //這裡拿it的值去算mean square error
    }
}



void update_weights(void)
{
    int i,j,k;

    for(i=0;i<num_layers-1;i++)
    {
        for(j=0;j<num_neurons[i];j++)
        {
            for(k=0;k<num_neurons[i+1];k++)
            {
                // Update Weights
                lay[i].neu[j].out_weights[k] = (lay[i].neu[j].out_weights[k]) - (alpha * lay[i].neu[j].dw[k]);
            }
            
            // Update Bias
            lay[i].neu[j].bias = lay[i].neu[j].bias - (alpha * lay[i].neu[j].dbias);
        }
    }   
}

void forward_prop(void)
{
    int i,j,k;

    for(i=1;i<num_layers;i++)
    {   
        for(j=0;j<num_neurons[i];j++)
        {
            lay[i].neu[j].z = lay[i].neu[j].bias;

            for(k=0;k<num_neurons[i-1];k++)
            {
                lay[i].neu[j].z  = lay[i].neu[j].z + ((lay[i-1].neu[k].out_weights[j])* (lay[i-1].neu[k].actv));
            }

            // Relu Activation Function for Hidden Layers
            if(i < num_layers-1)
            {
                if((lay[i].neu[j].z) < 0)
                {
                    lay[i].neu[j].actv = 0;
                }

                else
                {
                    lay[i].neu[j].actv = lay[i].neu[j].z;
                }
            }
            
            // Sigmoid Activation function for Output Layer
            else
            {
                lay[i].neu[j].actv = 1/(1+exp(-lay[i].neu[j].z));
                printf("Output: %d\n", (int)round(lay[i].neu[j].actv));
                //printf("\n");     不要這個換行讓底下的 Full Cost 可以跟在ouput下面
            }
        }
    }
}

// Compute Total Cost
void compute_cost(int i)
{
    int j;
    float tmpcost=0;
    float tcost=0;

    for(j=0;j<num_neurons[num_layers-1];j++)
    {
        tmpcost = desired_outputs[i][j] - lay[num_layers-1].neu[j].actv;
        cost[j] = (tmpcost * tmpcost)/2;
        tcost = tcost + cost[j];
        temp_MSE += tmpcost * tmpcost;          //算MSE用
    }   

    full_cost = (full_cost + tcost)/n;
    n++;
    printf("Full Cost: %f\n\n",full_cost);
}

// Back Propogate Error
void back_prop(int p)
{
    int i,j,k;

    // Output Layer
    for(j=0;j<num_neurons[num_layers-1];j++)
    {           
        lay[num_layers-1].neu[j].dz = (lay[num_layers-1].neu[j].actv - desired_outputs[p][j]) * (lay[num_layers-1].neu[j].actv) * (1- lay[num_layers-1].neu[j].actv);

        for(k=0;k<num_neurons[num_layers-2];k++)
        {   
            lay[num_layers-2].neu[k].dw[j] = (lay[num_layers-1].neu[j].dz * lay[num_layers-2].neu[k].actv);
            lay[num_layers-2].neu[k].dactv = lay[num_layers-2].neu[k].out_weights[j] * lay[num_layers-1].neu[j].dz;
        }
            
        lay[num_layers-1].neu[j].dbias = lay[num_layers-1].neu[j].dz;           
    }

    // Hidden Layers
    for(i=num_layers-2;i>0;i--)
    {
        for(j=0;j<num_neurons[i];j++)
        {
            if(lay[i].neu[j].z >= 0)
            {
                lay[i].neu[j].dz = lay[i].neu[j].dactv;
            }
            else
            {
                lay[i].neu[j].dz = 0;
            }

            for(k=0;k<num_neurons[i-1];k++)
            {
                lay[i-1].neu[k].dw[j] = lay[i].neu[j].dz * lay[i-1].neu[k].actv;    
                
                if(i>1)
                {
                    lay[i-1].neu[k].dactv = lay[i-1].neu[k].out_weights[j] * lay[i].neu[j].dz;
                }
            }

            lay[i].neu[j].dbias = lay[i].neu[j].dz;
        }
    }
}

// Test the trained network
void test_nn(void) 
{
    int i;
    while(1)
    {
        printf("\nEnter input to test:\n");           //因為我把output那行的一個\n刪掉了，所以我在Enter的前面加了\n

        for(i=0;i<num_neurons[0];i++)
        {
            scanf("%f",&lay[0].neu[i].actv);
        }
        forward_prop();
    }
}

// TODO: Add different Activation functions
//void activation_functions()

int dinit(void)
{
    // TODO:
    // Free up all the structures

    return SUCCESS_DINIT;
}



//算mean square error
void compute_MSE(int i)
{
    MSE = temp_MSE / (4 * (i + 1));
    fprintf(exfptr, "%d,%f\n" , i, MSE);
}


//感謝看完~ :)

