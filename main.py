import torch

from sparsifier import MLP

from torch.optim.lr_scheduler import LinearLR


def main():
    # Create a random tensor
    dense_emb = torch.nn.Embedding(20, 10)

    # Create a MLP
    mlp = MLP()

    # Define the loss function
    # loss = torch.nn.MSELoss()

    # Define the number of epochs
    nb_epochs = 50000

    # Define the optimizer
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=nb_epochs)

    # Set the MLP to training mode
    # mlp.train()

    # Train the MLP on the random tensor to preserve the cosine similarity
    # i.e cosine(dense[0], dense[1]) = cosine(output[0], output[1])

    dense_emb = dense_emb.weight

    for i in range(nb_epochs):
        # Forward pass
        sparse_emb = mlp(dense_emb)

        # Compute the cosine similarity
        # norm_dense = dense_emb / torch.norm(dense_emb, dim=1, keepdim=True)
        # dense_similarity = torch.mm(norm_dense, norm_dense.t())

        # sparse_emb = sparse_emb / torch.norm(sparse_emb, dim=1, keepdim=True)
        # sparse_similarity = torch.mm(sparse_emb, sparse_emb.t())

        # # Compute the cosine loss
        # cosine_loss = loss(dense_similarity, sparse_similarity)

        # Compute the L1 regularization
        # l1_reg = sum(torch.norm(p, 1) for p in mlp.parameters())
        l1_reg = torch.norm(sparse_emb, p=1)
        # l1_reg = l1_reg + sum(torch.norm(p, 1) for p in mlp.parameters())

        # Compute the total loss
        # output = 0 * cosine_loss + 0.01 * l1_reg
        output = l1_reg

        # Backward pass
        output.backward()

        # Update the weights
        optimizer.step()
        scheduler.step()

        # Zero the gradients
        optimizer.zero_grad()

        if i % 50 == 0:
            # Print the loss with the sparsity
            sparsity = torch.mean((sparse_emb == 0).float())
            small = torch.mean((sparse_emb < 1e-8).float())
            # print(
            #     f"Cosine loss: {cosine_loss.item():.2f} L1 reg: {l1_reg.item():.2f} Embeddings Sparsity: {sparsity.item():.2f} Weights Sparsity: {mlp.avg_zeros():.2f}"
            # )
            print(
                f"L1 reg: {l1_reg.item()} Sparsity: {sparsity.item():.2f} Small: {small.item():.2f} LR {scheduler.get_last_lr()[0]}"
            )

    # Set the MLP to evaluation mode
    mlp.eval()

    print("Done!")


if __name__ == "__main__":
    main()
