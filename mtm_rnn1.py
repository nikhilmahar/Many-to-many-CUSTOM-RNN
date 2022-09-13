# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 17:36:36 2022

@author: NIKHIL
"""
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt 
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
x=np.linspace(0,1,50)
y=np.sin(30*x)
plt.figure()
plt.plot(x,y)
plt.show()

input_signal=torch.tensor(x)
input_signal=input_signal.reshape(1,len(input_signal))
output_signal=torch.tensor(y)
output_signal=output_signal.reshape(len(output_signal),1)

class RNN(nn.Module):
    # implement RNN from scratch rather than using nn.RNN
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2h=  nn.Linear(hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.h2o=nn.Linear(hidden_size, output_size)
        
        self.tanh = nn.Sigmoid()
        
    def forward(self, input_tensor, hidden_tensor):
        #print("it",input_tensor.shape,"ht",hidden_tensor.shape)
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        
       # print("combined",combined.shape)
        
        hidden = self.i2h(combined)
        hidden = self.tanh(hidden)
        hidden = self.h2h(hidden)
        hidden = self.tanh(hidden)
        hidden = self.h2h(hidden)
        hidden = self.tanh(hidden)
        hidden = self.h2h(hidden)
        hidden = self.tanh(hidden)
        hidden = self.h2h(hidden)
        hidden = self.tanh(hidden)
        hidden = self.h2h(hidden)
        #print("hidden vector",hidden.shape)
        output = self.i2h(combined)
        output = self.tanh(output)
        output = self.h2h(output)
        output = self.tanh(output)
        output = self.h2h(output)
        output = self.tanh(output)
        output = self.h2h(output)
        output = self.tanh(output)
        output = self.h2o(output)
        
       # print("output vector",output.shape)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
hidden_size = 100
input_size=1
output_size=1

rnn = RNN(input_size, hidden_size, output_size)

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
    print("epochs:",i, "loss",loss.item())
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
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
print("predicted output",output,"orignal output", output_signal)  
plt.figure(2)      
plt.plot(input_tensor.T.detach().numpy(),output.detach().numpy(),"r",label="pred")
plt.plot(x,y,"--k",label="true")
plt.legend(loc = 'upper center',facecolor = 'w')
plt.show()