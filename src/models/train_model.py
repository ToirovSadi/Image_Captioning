from tqdm import tqdm
import torch

def train_epoch(
    model,
    train_dataloader,
    loss_fn,
    optimizer,
    epoch=None,
    clip=None,
):
    loop = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f"Training: {'' if epoch is None else epoch}"
    )
    model.train()
    total_loss = 0
    for i, (images, captions) in loop:
        optimizer.zero_grad()
        
        preds = model(images, captions[:, :-1])
        
        # captions.shape: [batch_size, sent_size]
        # preds.shape: [batch_size, max_sent_size, output_dim]
        output_dim = model.decoder.output_dim
        preds = preds.reshape(-1, output_dim)
        captions = captions[:, 1:].reshape(-1)

        loss = loss_fn(preds, captions)
        
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(**{'loss': total_loss / (i + 1)})
    return total_loss / len(train_dataloader)
        
def eval_epoch(
    model,
    val_dataloader,
    loss_fn,
    epoch=None,
):
    loop = tqdm(
        enumerate(val_dataloader),
        total=len(val_dataloader),
        desc=f"Validation: {'' if epoch is None else epoch}"
    )
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (images, captions) in loop:
            preds = model(images, captions[:, :-1])

            # captions.shape: [batch_size, sent_size]
            # preds.shape: [batch_size, max_sent_size, output_dim]
            output_dim = model.decoder.output_dim
            preds = preds.reshape(-1, output_dim)
            captions = captions[:, 1:].reshape(-1)

            loss = loss_fn(preds, captions)

            total_loss += loss.item()
            loop.set_postfix(**{'loss': total_loss / (i + 1)})
            
    return total_loss / len(val_dataloader)

def train(
    model,
    epochs,
    dataloaders,
    loss_fn,
    optimizer,
    grad_clip,
    ckpt_path='best.pt',
    best_so_far=float('inf'),
):
    for epoch in range(1, epochs+1):
        train_loss = train_epoch(model, dataloaders[0], loss_fn, optimizer, epoch, grad_clip)
        if len(dataloaders) > 1:
            val_loss = eval_epoch(model, dataloaders[1], loss_fn, epoch)
        else:
            val_loss = train_loss
        
        if val_loss < best_so_far:
            best_so_far = val_loss
            torch.save(model.state_dict(), ckpt_path)
        
    return best_so_far