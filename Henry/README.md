# Pytorch Training loop

```python
def fit():
    for epoch in range(epochs):
        for xb,yb in train_dl:         # iterate from a dataloader (mini-batchs)
            pred = model(xb)           # calculate predictions
            loss = loss_func(pred, yb) # calculate loss
            loss.backward()            # Calculate gradients
            opt.step()                 # Update with the learning rate
            opt.zero_grad()            # Reset gradient (Different from tensorflow)
```
