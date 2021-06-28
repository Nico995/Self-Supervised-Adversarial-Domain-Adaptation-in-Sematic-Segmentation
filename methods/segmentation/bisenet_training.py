from torch.cuda.amp import autocast


def bisenet_training(model, data, label, optimizer, scaler, criterion, loss):
    # Set model to Train mode
    model.train()

    # Clear optimizer gradient in an efficient way
    optimizer.zero_grad(set_to_none=True)

    with autocast():
        # Get network output
        output, output_sup1, output_sup2 = model(data)

        # Loss
        loss1 = criterion(output, label)
        loss2 = criterion(output_sup1, label)
        loss3 = criterion(output_sup2, label)
        loss = loss1 + loss2 + loss3

    # Compute gradients with gradient scaler
    scaler.scale(loss).backward()

    scaler.step(optimizer)
    # Updates the scale for next iteration.
    scaler.update()

    return loss.item()
