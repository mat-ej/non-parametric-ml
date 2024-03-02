# %%
import numpy as np
import matplotlib.pyplot as plt
import torch

# Global parameter to control plot display
show_plots = True

np.random.seed(42)
n = 50
x = np.array(np.random.randn(n), dtype=np.float32)
y = np.array(
  0.75 * x**2 + 1.0 * x + 2.0 + 0.3 * np.random.randn(n),
  dtype=np.float32)

# plt.scatter(x, y, facecolors='none', edgecolors='b')
# plt.scatter(x, y, c='r')
# plt.show()

# %%
model = torch.nn.Linear(1,1)
model.weight.data.fill_(6.0)
model.bias.data.fill_(-3.0)
models = [[model.weight.item(), model.bias.item()]]


loss_fn = torch.nn.MSELoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# NOTE: requires_grad_ is a setter while requires_grad is property
def numpy_to_torch(x, y):
    inputs = torch.from_numpy(x).requires_grad_(True).reshape(-1, 1)
    labels = torch.from_numpy(y).reshape(-1, 1)
    return inputs, labels

for epoch in range(100):
    inputs, labels = numpy_to_torch(x, y)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    models.append([model.weight.item(), model.bias.item()])
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: loss = {loss.item()}')

# %%

weight = model.weight.item()
bias = model.bias.item()
print(f'weight = {weight}, bias = {bias}')

plt.scatter(x, y)
plt.plot(x, weight * x + bias, 'r')
plt.show()

# print()
# %%
def get_loss_map(loss_fn, x, y):
    num_steps = 101  # Number of steps in the grid
    w_start, w_end = -5.0, 8.0  # Start and end values for weights
    b_start, b_end = -5.0, 8.0  # Start and end values for biases
    
    # Initialize a 2D list to store loss values
    losses = [[0.] * num_steps for _ in range(num_steps)]
    
    # Convert numpy arrays to torch tensors
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    # Generate weight (w) and bias (b) values and compute loss
    for wi in range(num_steps):
        for bi in range(num_steps):
            # Linearly interpolate weight and bias values
            w = w_start + (w_end - w_start) * wi / (num_steps - 1)
            b = b_start + (b_end - b_start) * bi / (num_steps - 1)
            
            # Compute model output using the interpolated weight and bias
            y_pred = x * w + b
            
            # Calculate and store the loss
            losses[wi][bi] = loss_fn(y_pred, y).item()
    
    # Reverse the list of losses for plotting purposes
    return list(reversed(losses))

# %%
import pylab
loss_fn = torch.nn.MSELoss()
losses = get_loss_map(loss_fn, x, y)
cm = pylab.get_cmap('terrain')

fig, ax = plt.subplots()
plt.xlabel('Bias')
plt.ylabel('Weight')
i = ax.imshow(losses, cmap=cm, interpolation='nearest', extent=[-5, 8, -5, 8])
fig.colorbar(i)
plt.show()

# %%
cm = pylab.get_cmap('terrain')
fig, ax = plt.subplots()
plt.xlabel('Bias')
plt.ylabel('Weight')
i = ax.imshow(losses, cmap=cm, interpolation='nearest', extent=[-5, 8, -5, 8])

model_weights, model_biases = zip(*models)
ax.scatter(model_biases, model_weights, c='r', marker='+')
ax.plot(model_biases, model_weights, c='r')

fig.colorbar(i)
plt.show()

# %%
def learn(loss_fn, x, y, lr=0.1, epochs=100, momentum=0, weight_decay=0, dampening=0, nesterov=False):
    '''
    Trains a linear model using the specified loss function and optimizer parameters.
    '''
    # Initialize model
    model = torch.nn.Linear(1, 1)
    model.weight.data.fill_(6.0)  # Initialize weight
    model.bias.data.fill_(-3.0)  # Initialize bias

    # Setup optimizer with given parameters
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, 
                                weight_decay=weight_decay, dampening=dampening, nesterov=nesterov)

    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x).reshape(-1, 1)
    labels = torch.from_numpy(y).reshape(-1, 1)

    models = []  # To store model parameters at each epoch

    for epoch in range(epochs):
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = loss_fn(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters

        # Store model parameters
        models.append([model.weight.item(), model.bias.item()])

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: loss = {loss.item()}')

    return model, models

# %%
def multi_plot(lr=0.1, epochs=100, momentum=0, weight_decay=0, dampening=0, nesterov=False):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
  for loss_fn, title, ax in [
    (torch.nn.MSELoss(), 'MSELoss', ax1),
    (torch.nn.L1Loss(), 'L1Loss', ax2),
    (torch.nn.HuberLoss(), 'HuberLoss', ax3),
    (torch.nn.SmoothL1Loss(), 'SmoothL1Loss', ax4),
  ]:
    losses = get_loss_map(loss_fn, x, y)
    model, models = learn(
      loss_fn, x, y, lr=lr, epochs=epochs, momentum=momentum,
      weight_decay=weight_decay, dampening=dampening, nesterov=nesterov)

    cm = pylab.get_cmap('terrain')
    i = ax.imshow(losses, cmap=cm, interpolation='nearest', extent=[-5, 8, -5, 8])
    ax.title.set_text(title)
    loss_w, loss_b = zip(*models)
    ax.scatter(loss_b, loss_w, c='r', marker='+')
    ax.plot(loss_b, loss_w, c='r')

  plt.show()

multi_plot(lr=0.1, epochs=100)

# %%

'''
momentum, which dictates how much of the last step’s gradient to add 
in to the current gradient update going froward.
'''


multi_plot(lr=0.1, epochs=100, momentum=0.1)
# %%

'''
-Normal momentum adds in some of the gradient from the last step to the gradient for the current step, giving us the scenario in figure 7(a) below. But if we already know where the gradient from the last step is going to carry us, then 
-Nesterov momentum instead calculates the current gradient by looking ahead to where that will be
'''


multi_plot(lr=0.1, epochs=100, momentum=0.9, nesterov=True)

# %%

'''
weight decay adds a regularizing L2 penalty on the values of the parameters
-effectively offsets the optimum from the rightful optima and closer to (0, 0)
-least pronounced in L2 loss, loss vals are large enough to offsert L2 penalties on the weights
'''

multi_plot(lr=0.1, epochs=100, momentum=0.9, nesterov=True, weight_decay=2.0)

# %%

'''
Dampening, which discounts the momentum by the dampening factor
'''

multi_plot(lr=0.1, epochs=100, momentum=0.9, dampening=0.8)
