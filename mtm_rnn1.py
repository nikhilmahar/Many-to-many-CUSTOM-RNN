import torch
import torch.nn as nn 
import scipy as sc
import numpy as np
import sympy as sym
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sympy import init_printing
init_printing()

import matplotlib.pyplot as plt 
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# =============================================================================
# response generation
# =============================================================================
from sympy import init_printing
init_printing()

# Define the system parameters
m, k, c, t = sym.symbols('m, k, c, t')

# Define the main 
x = sym.Function('x')
lhs = m * x(t).diff(t,2) + k * x(t) + c * x(t).diff(t,1)
eq_main = sym.Eq(lhs, 0)
eq_main
eq_acc = sym.solve(eq_main, x(t).diff(t,2))[0]
eq_acc
sym.Eq(x(t).diff(t,2), sym.expand(eq_acc))
def SDOF_system(y, t, m, k, c):
    x, dx = y
    dydt = [dx,
           -c/m*dx - k/m*x]
    return dydt
m =1 # mass in kg
k= 150 # Spring Stiffnes N/m
c = 0.6# Dampening in Ns/m
y0 = [1.0, 0.0]

t = np.linspace(0, 5, 200)
from scipy.integrate import odeint 
sol = odeint(SDOF_system, y0, t, args=(m, k, c))
y=sol[:,0]
plt.figure(figsize=(10,5))
plt.plot(t, sol[:, 0], 'b', label='x(t)')
plt.legend(loc='best')
plt.xlabel('t (sec)')
plt.grid()

plt.show()
# =============================================================================
# 
# =============================================================================

input_signal=torch.tensor(t)
input_signal=input_signal.reshape(1,len(input_signal))
output_signal=torch.tensor(y)
output_signal=output_signal.reshape(len(output_signal),1)

class RNN(nn.Module):
    # implement RNN from scratch rather than using nn.RNN
    def __init__(self, input_size, hidden_size0,hidden_size1,hidden_size2,hidden_size3,hidden_size4,output_size):
        super(RNN, self).__init__()
        self.hidden_size0 = hidden_size0
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.i2h1 = nn.Linear(input_size + hidden_size0, hidden_size1)
        self.h12h2=  nn.Linear(hidden_size1, hidden_size2)
        self.h22h3=  nn.Linear(hidden_size2, hidden_size3)
        self.h32h4=  nn.Linear(hidden_size3, hidden_size4)
        
        self.i2o = nn.Linear(input_size + hidden_size0, output_size)
        #self.h2o=nn.Linear(hidden_size, output_size)
        
        self.tanh = nn.Tanh()
        
    def forward(self, input_tensor, hidden_tensor):
        #print("it",input_tensor.shape,"ht",hidden_tensor.shape)
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        
       # print("combined",combined.shape)
        
        hidden = self.i2h1(combined)
        hidden = self.tanh(hidden)
        hidden = self.h12h2(hidden)
        hidden = self.tanh(hidden)
        hidden = self.h22h3(hidden)
        hidden = self.tanh(hidden)
        hidden = self.h32h4(hidden)
        hidden = self.tanh(hidden)
        # hidden = self.h2h(hidden)
        # hidden = self.tanh(hidden)
        # hidden = self.h2h(hidden)
        #print("hidden vector",hidden.shape)
        output = self.i2o(combined)
        #output = self.tanh(output)
        #output = self.h2h(output)
        # output = self.tanh(output)
        # output = self.h2h(output)
        # output = self.tanh(output)
        # output = self.h2h(output)
        # output = self.tanh(output)
        # output = self.h2o(output)
        
       # print("output vector",output.shape)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size0)
hidden_size0=32    
hidden_size1 = 128
hidden_size2 = 256
hidden_size3 = 128
hidden_size4 = 32
input_size=1
output_size=1

rnn = RNN(input_size,hidden_size0, hidden_size1,hidden_size2,hidden_size3,hidden_size4, output_size)

# =============================================================================
# # one step
# input_tensor=input_signal[:,0]
# input_tensor=input_tensor.reshape(1,1)
# hidden_tensor = rnn.init_hidden()
# 
# output, next_hidden = rnn(input_tensor, hidden_tensor)
# print(output.size())
# print(next_hidden.size())
# =============================================================================
input_tensor=input_signal

out=torch.zeros(input_tensor.size()[1],1)
learning_rate = 0.001
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)


def train(input_tensor, output_signal):
    hidden = rnn.init_hidden()
    loss=torch.zeros(input_tensor.size()[1],1)
    output= torch.zeros(input_tensor.size()[1],1)
    
    for i in range(input_tensor.size()[1]):
        inp_tensor=input_tensor[:,i].reshape(1,1)
        pred_output, hidden = rnn(inp_tensor, hidden)
        #print("pred_out",pred_output.shape)
        loss[i]=(pred_output-output_signal[i,:])**2
        
        output[i]=pred_output
        #out.append(pred_output)
        #out=pred_output
        #print("i",i,"out",len(out))
    #output = torch.cat(out) 
    #print("i",i,"output:",j,output.shape)
    loss1=torch.mean(loss)
    #loss = torch.mean((output[j:j+4,:].squeeze()-output_signal)**2)
      
       
#        loss=current_loss
    #print("out",out)
    #loss=torch.mean((out-output_signal)**2)
    
    
    return output, loss1



n_iters = 50000

for i in range(n_iters):
    #print("i",i,"j",j)
    #first_input= input_tensor[:,0]
    #first_input=first_input.reshape(1,1)
    output, loss = train(input_tensor.float(), output_signal.float())
    
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    if i%100==0:
        print("epochs:",i+1, "loss",loss.item())
        plt.figure(2)      
        plt.plot(input_tensor.T.detach().numpy(),output.detach().numpy(),"r",label="pred")
        plt.plot(t,y,"--k",label="true")
        plt.title("solution",)
        plt.legend(loc = 'upper right',facecolor = 'w')
        plt.text(1.065,0.7,"Epoch: %i"%(i+1),fontsize="xx-large",color="k")
  
        plt.show()

    # j=j+5
    # print("j",j)
    #current_loss += loss 
    
#     if (i+1) % plot_steps == 0:
#         all_losses.append(current_loss / plot_steps)
#         current_loss = 0
        
#     if (i+1) % print_steps == 0:
#         guess = category_from_output(output)
#         correct = "CORRECT" if guess == category else f"WRONG ({category})"
#         print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}")
# print("predicted output",output,"orignal output", output_signal)  
# plt.figure(2)      
# plt.plot(input_tensor.T.detach().numpy(),output.detach().numpy(),"r",label="pred")
# plt.plot(t,y,"--k",label="true")
# plt.legend(loc = 'upper center',facecolor = 'w')
# plt.show()
