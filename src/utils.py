
import torch
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def list_of_distances(X, Y):
    # both squared norms
    X = X.to(device)
    Y = Y.to(device)
    X_norm = torch.unsqueeze(torch.sum(torch.pow(X, 2), dim=1), 1)
    Y_norm = torch.sum(torch.pow(Y, 2), dim=-1)
    
    return X_norm + Y_norm - 2*torch.matmul(X, torch.transpose(Y, 0, -1))

if __name__ == "__main__":

    # test
    protoypes = 3
    latent_dims = 4
    batch_size = 4

    X = torch.randn(batch_size, latent_dims)
    Y = torch.randn(protoypes, latent_dims)

    print(list_of_distances(X, Y))

    # numpy distance veriication
    for x in X:
        distances = []
        for y in Y:
            distances.append(round(np.linalg.norm(x - y)**2, 4))
        print(distances)
