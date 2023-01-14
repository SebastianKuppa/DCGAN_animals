import torch.cuda
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from cnn_utils import AnimalDataset, Discriminator, Generator

batch_size = 32
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0002

if __name__ == '__main__':
    # init loss function
    loss_function = nn.BCELoss()

    netD = Discriminator(ngpu=0)
    netG = Generator(ngpu=0)

    # init optimizer for Discriminator and Generator
    d_optim = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optim = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    train_set = AnimalDataset(root_dir='D:/datasets/afhq/afhq/train/cat/')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for idx, (imgs,) in enumerate(train_loader):
            # training the discriminator
            ###########################
            # take images from dataset and feed to discriminator
            real_inputs = imgs.to(device)
            real_outputs = netD(real_inputs)
            # create labels with value 1=real images
            real_label = torch.ones(real_inputs.shape[0], 1).to(device)

            # create input noice tensor for generator
            noise = (torch.rand(real_inputs.shape[0], 100, 1, 1) - 0.5) / 0.5
            noise = noise.to(device)

            fake_inputs = netG(noise)
            fake_outputs = netD(fake_inputs)
            fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

            outputs = torch.cat((real_outputs.view(-1).unsqueeze(1), fake_outputs.view(-1).unsqueeze(1)),
                                dim=0)
            targets = torch.cat((real_label, fake_label), dim=0)

            d_loss = loss_function(outputs, targets)
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # training the generator
            ###########################
            # the goal of the generator is to make the discriminator believe every image should be
            # labelled as 1
            fake_outputs = netD(fake_inputs)
            fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
            g_loss = loss_function(fake_outputs.view(-1).unsqueeze(1), fake_targets)
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            if idx % 5 == 0 or idx == len(train_loader):
                print(f'Epoch {epoch} Iteration {idx}: discriminator loss {d_loss.item():.3f} '
                      f'generator loss {g_loss.item():.3f} ')
