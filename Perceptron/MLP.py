from matplotlib.pyplot import pause, plot
import numpy as np
import matplotlib.pyplot as plt 

def rand(start,end,shape):
    size = shape[0]*shape[1]
    a = np.linspace(start=start,stop=end,num=size)
    r = a.reshape(shape)
    return r

class MLP():
    def __init__(self,input,num_neurons_hidden_layer,learning_rate=0.5):
        self.input = np.array(input)
        self.input_weights = rand(-0.5,0.5,shape=(self.input.shape[1]+1,num_neurons_hidden_layer)) ##* initialize first layer weights
        self.output_weights = rand(-0.5,0.5,shape=(num_neurons_hidden_layer+1,self.input.shape[1])) ##* initialize secound layer weights
        self.alpha = learning_rate
        self.num_neurons_hidden_layer = num_neurons_hidden_layer
        self.learning_rate = learning_rate

    def activation(self,x_array_input):
        result = x_array_input
        r = 0
        for i in x_array_input:
            l =0
            for j in i:
                power = np.exp(-j)
                result[r][l] = ((1-power)/(1+power))
                l += 1
            r += 1    
        return result

    def one_d_activation(self,x_array_input):
        power = np.exp(x_array_input) 
        result = ((1-power)/(1+power))
        return result

        
    def differential_activation(self,_in,coefficient=0.5):
        f_in = self.activation(_in)
        return (coefficient*(1+f_in)*(1-f_in))

    def differential_one_d_activation(self,_in,coefficient=0.5):
        f_in = self.one_d_activation(_in)
        return (coefficient*(1+f_in)*(1-f_in))

    def trainnig(self,target=None,epsilon=0.5,velocity_trainnig=0.02):
        epoch = 0
        flage = True
        while(flage):            
            flage = False
            Max = 0
            epoch += 1
            #####*(1)feedforward
            for i in range(len(self.input)):
                z_in = self.input[i]@self.input_weights[1:] + self.input_weights[0]
                z = self.one_d_activation(z_in)
                y_in = z@self.output_weights[1:] + self.output_weights[0]
                y = self.one_d_activation(y_in) 
            #####*(2)Backpropagation of error
                delta_k = (target[i] - y)*self.differential_one_d_activation(y_in).reshape(1,len(target[i]))
                z_t = np.transpose(z).reshape(len(z),1)
                delta_w = self.learning_rate*(z_t)@delta_k
                delta_w = np.append(delta_w,self.learning_rate*delta_k,axis=0)  ## bias appended for out put weights
                delta_in = delta_k@(np.transpose(self.output_weights))
                delta = np.delete(delta_in,0)
                delta = (delta*self.differential_one_d_activation(z_in)).reshape(1,len(z_in))
                x_input = np.transpose(self.input[i]).reshape(len(self.input[i]),1)
                delta_v = self.learning_rate*(x_input@delta)
                delta_v = np.append(delta_v,self.learning_rate*delta,axis=0) ## bias appended for out put weights
            #####*(3)Update weights and biases     
                self.output_weights += delta_w
                self.input_weights += delta_v                 
            #####! Stopping condition
                max_w_change = np.amax(max(np.amax(np.abs(delta_v)) , np.amax(np.abs(delta_w))))
                if max_w_change > Max:
                    Max = max_w_change
            if Max > epsilon:
                flage = True

            plt.plot(self.input,'o',color='green') 
            plt.plot(self.output_weights,'D',color='red')        
            plt.plot(self.input_weights,'x',color='blue')    
            plt.plot(y,'o',color='black') 
            pause(velocity_trainnig)
            print(epoch)
        print(epoch)    
        
        return y
    def use(self):
        pass 


data = np.loadtxt('font.txt',delimiter=',',dtype=float)
p = MLP(data,num_neurons_hidden_layer=20,learning_rate=0.02) 
p.trainnig(data,epsilon=0.025,velocity_trainnig=0.02)

plt.show()