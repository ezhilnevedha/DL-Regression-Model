# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:EZHIL NEVEDHA K

### Register Number:212223230055

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

X = torch.linspace(1,70,70).reshape(-1,1)
torch.manual_seed(71)
e = torch.randint(-8,9,(70,1),dtype=torch.float)
e

y = 2*X + 1 + e
print(y.shape)

plt.scatter(X.numpy(), y.numpy(),color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()


torch.manual_seed(59)

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear=nn.Linear(in_features,out_features)
    def forward(self,x):
      return self.linear(x)

torch.manual_seed(59)
model=Model(1,1)

initial_weight=model.linear.weight.item()
initial_bias=model.linear.bias.item()

print("\nName: EZHIL NEVEDHA K")
print("\nRegister Number: 212223230055")

print(f"\nInitial Weight: {initial_weight:.8f}, \nInitial Bias: {initial_bias:.8f}")
loss_function=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
epochs=100
losses=[]
for epoch in range(1,epochs+1):
  optimizer.zero_grad()
  y_pred=model(X)
  loss=loss_function(y_pred,y)
  losses.append(loss.item())
  loss.backward()
  optimizer.step()
  print(f"epoch: {epoch:2} loss: {loss.item():10.8f}"
         f"weight: {model.linear.weight.item():10.8f}"
         f"bias: {model.linear.bias.item():10.8f}")


plt.plot(range(epochs),losses,color="Blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()

final_weight=model.linear.weight.item()
final_bias=model.linear.bias.item()

print("\nName: EZHIL NEVEDHA K")
print("\nRegister Number: 212223230055")

print(f"\nFinal Weight: {final_weight:.8f}, Final Bias: {final_bias:.8f}")
x1=torch.tensor([X.min().item(), X.max().item()])
y1=x1*final_weight+final_bias
print(x1)
print(y1)

plt.scatter(X.numpy(), y.numpy(), color='red', label='Original Data')
plt.plot(x1.numpy(), y1.numpy(), color='cyan', label='Best-Fit Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()

x_new=torch.tensor([[120.0]])
y_new_pred=model(x_new).item()

print("\nName: EZHIL NEVEDHA K")
print("\nRegister Number: 212223230055")

print(f"\nPredicted for x = 120: {y_new_pred:.8f}")
```

### Dataset Information
<img width="747" height="515" alt="image" src="https://github.com/user-attachments/assets/59eac139-d780-45eb-9985-8e1e02573aeb" />


### OUTPUT
Training Loss Vs Iteration Plot

<img width="580" height="453" alt="download" src="https://github.com/user-attachments/assets/22c89505-f4da-4655-adb0-8d0e7f544cda" />

Best Fit line plot

<img width="571" height="453" alt="download" src="https://github.com/user-attachments/assets/3a4ec2b8-cda0-4b33-8433-934faee7037f" />

### New Sample Data Prediction
<img width="460" height="288" alt="image" src="https://github.com/user-attachments/assets/387ab514-29e5-4bde-8754-e2fd3fb1f5f4" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
