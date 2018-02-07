import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision
import itertools
from torchvision.utils import save_image


class Encoder(nn.Module):
    def __init__(self, in_channels, num_filters, num_layers, z_size, is_training=False):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.is_training = is_training
        conv_list = []
        for i in xrange(num_layers):
            scale = 2 ** (num_layers - i - 1)
            conv_i = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels, out_channels=num_filters / scale,
                    kernel_size=4, stride=2, padding=2
                ),
                nn.BatchNorm2d(num_features=num_filters / scale),
                nn.ReLU(inplace=True)
            )
            conv_list.append(conv_i)
            self.in_channels = num_filters / scale
        self.conv = nn.Sequential(*conv_list)
        self.linear = nn.Linear(in_features=3 * 3 * num_filters, out_features=z_size)
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0099999)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        if self.is_training:
            noise = torch.normal(torch.zeros_like(x.data), std=0.01)
            x.data += noise
        conv_out = self.conv(x).view(-1, 3 * 3 * 1024)
        z = self.linear(conv_out)
        return z


class Decoder(nn.Module):
    def __init__(self, num_filters, num_layers, z_size, output_shape):
        super(Decoder, self).__init__()
        height = output_shape / 2 ** (num_layers - 1) + 1
        width = output_shape / 2 ** (num_layers - 1) + 1
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=z_size, out_features=num_filters * height * width),
            nn.ReLU(inplace=True)
        )
        self.in_channels = num_filters
        deconv_list = []
        for i in xrange(num_layers - 1):
            scale = 2 ** (i + 1)
            deconv_i = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels, out_channels=num_filters / scale,
                    kernel_size=4, stride=2, padding=2, output_padding=1
                ),
                nn.BatchNorm2d(num_features=num_filters / scale),
                nn.ReLU(inplace=True)
            )
            deconv_list.append(deconv_i)
            self.in_channels = num_filters / scale
        self.deconv = nn.Sequential(*deconv_list)
        self.deconv_last = nn.ConvTranspose2d(
            in_channels=self.in_channels, out_channels=1,
            kernel_size=4, stride=1, padding=2
        )
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0099999)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()

    def forward(self, z):
        linear_out = self.linear1(z)
        deconv_input = linear_out.view(-1, 1024, 8, 8)
        deconv_out = self.deconv(deconv_input)
        recon_x = self.deconv_last(deconv_out)
        return F.sigmoid(recon_x), recon_x


class Adversary_z(nn.Module):
    def __init__(self, num_filters, num_layers, z_size):
        super(Adversary_z, self).__init__()
        self.in_features = z_size
        linears = []
        for i in xrange(num_layers):
            linear_i = nn.Sequential(
                nn.Linear(in_features=self.in_features, out_features=num_filters),
                nn.ReLU(inplace=True)
            )
            linears.append(linear_i)
            self.in_features = num_filters
        self.linear = nn.Sequential(*linears)
        self.final_linear = nn.Linear(in_features=self.in_features, out_features=1)
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0099999)
                m.bias.data.zero_()

    def forward(self, z):
        linear_out = self.linear(z)
        out = self.final_linear(linear_out)
        return out


def run():
    # define the hyper-parameters
    batch_size = 100
    e_pretrain_batch_size = 1000
    pretrain_epochs = 200
    epochs = 100
    z_size = 8
    lam = 10

    e_num_filters = 1024
    e_num_layers = 4

    g_num_filters = 1024
    g_num_layers = 3

    d_num_filters = 512
    d_num_layers = 4

    # download and load data
    train_data = MNIST(
        root='data/',
        train=True,
        transform=torchvision.transforms.ToTensor()
    )
    test_data = MNIST(
        root='data/',
        train=False,
        transform=torchvision.transforms.ToTensor()
    )
    # get the data batch
    pretrain_data_loader = DataLoader(dataset=train_data, batch_size=e_pretrain_batch_size, shuffle=True)
    pretest_data_loader = DataLoader(dataset=test_data, batch_size=e_pretrain_batch_size, shuffle=False)
    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # models
    encoder = Encoder(
        in_channels=1, num_filters=e_num_filters, num_layers=e_num_layers, z_size=z_size, is_training=True
    )
    decoder = Decoder(
        num_filters=g_num_filters, num_layers=g_num_layers, z_size=z_size, output_shape=28
    )
    discriminator = Adversary_z(num_filters=d_num_filters, num_layers=d_num_layers, z_size=z_size)

    # load model parameters
    # encoder.load_state_dict(torch.load('Pretrain_Encoder_epoch200.pth'))
    # encoder.load_state_dict(torch.load('encoder_100.pth'))
    # decoder.load_state_dict(torch.load('decoder_100.pth'))
    # discriminator.load_state_dict(torch.load('discriminator_100.pth'))

    if torch.cuda.is_available():
        encoder, decoder, discriminator = encoder.cuda(), decoder.cuda(), discriminator.cuda()

    # define the optimizer
    e_params = encoder.parameters()
    ae_params = itertools.chain(encoder.parameters(), decoder.parameters())
    d_params = discriminator.parameters()
    optimizer_e = optim.Adam(e_params, lr=1e-03, betas=(0.5, 0.999))
    optimizer_ae = optim.Adam(ae_params, lr=1e-03, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(d_params, lr=5e-04, betas=(0.5, 0.999))

    # pretrain model
    def sample_pz(batch_size=100, z_size=8):
        return Variable(torch.normal(torch.zeros(batch_size, z_size), std=1).cuda())

    def pretrain_loss(encoded, sample_noise):
        # for mean
        mean_qz = torch.mean(encoded, dim=0, keepdim=True)
        mean_pz = torch.mean(sample_noise, dim=0, keepdim=True)
        mean_loss = F.mse_loss(mean_qz, mean_pz)

        # for covariance
        cov_qz = torch.matmul((encoded - mean_qz).transpose(0, 1), encoded - mean_qz)
        cov_qz /= e_pretrain_batch_size - 1.
        cov_pz = torch.matmul((sample_noise - mean_pz).transpose(0, 1), sample_noise - mean_pz)
        cov_pz /= e_pretrain_batch_size - 1.
        cov_loss = F.mse_loss(cov_qz, cov_pz)

        return mean_loss + cov_loss

    def encoder_pretrain(epoch):
        encoder.train()
        for batch_idx, (data, _) in enumerate(pretrain_data_loader):
            if torch.cuda.is_available():
                data = data.cuda()
            data = Variable(data)
            sample_noise = sample_pz(e_pretrain_batch_size, z_size)
            encoded = encoder(data)

            optimizer_e.zero_grad()
            loss_pretrain = pretrain_loss(encoded, sample_noise)
            loss_pretrain.backward()
            optimizer_e.step()
            if batch_idx % 10 == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(pretrain_data_loader.dataset),
                           100. * batch_idx / len(pretrain_data_loader), loss_pretrain.data[0]
                ))
        # save the pretrain model at the last epoch
        if epoch == 200:
            torch.save(encoder.state_dict(), 'Pretrain_Encoder_epoch{:02d}.pth'.format(epoch))

    # train models
    d_loss_function = nn.BCEWithLogitsLoss()

    def gan_loss(sample_qz, sample_pz):
        logits_qz = discriminator(sample_qz)
        logits_pz = discriminator(sample_pz)

        # losses
        loss_qz = d_loss_function(logits_qz, torch.zeros_like(logits_qz))
        loss_pz = d_loss_function(logits_pz, torch.ones_like(logits_pz))
        loss_qz_trick = d_loss_function(logits_qz, torch.ones_like(logits_qz))
        loss_adversary = lam * (loss_qz + loss_pz)
        loss_penalty = loss_qz_trick
        return (loss_adversary, logits_qz, logits_pz), loss_penalty

    def train(epoch):
        for param_group in optimizer_ae.param_groups:
            print(param_group['lr'], "learning rate for Auto-Encoder.")
        for param_group in optimizer_d.param_groups:
            print(param_group['lr'], "learning rate for Discriminator.")
        encoder.train(), decoder.train(), discriminator.train()
        for batch_idx, (data, _) in enumerate(train_data_loader):
            if torch.cuda.is_available():
                data = data.cuda()
            data = Variable(data)
            sample_noise = sample_pz(batch_size, z_size)

            encoded = encoder(data)
            # for reconstructed
            recon_x, recon_logits = decoder(encoded)
            # for sample
            decoded, decoded_logits = decoder(sample_noise)

            # losses
            recon_loss = F.mse_loss(recon_x, data)
            loss_gan, loss_penalty = gan_loss(encoded, sample_noise)
            loss_wae = recon_loss + lam * loss_penalty
            loss_adv = loss_gan[0]

            # optimize wae
            encoder.zero_grad()
            decoder.zero_grad()
            loss_wae.backward(retain_graph=True)
            optimizer_ae.step()

            # optimize adv
            discriminator.zero_grad()
            loss_adv.backward()
            optimizer_d.step()

            if batch_idx % 10 == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)] \tWAE_Loss: {:.6f}\tD_Loss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_data_loader.dataset),
                           100. * batch_idx / len(train_data_loader), loss_wae.data[0], loss_adv.data[0]
                ))
        # save images and save models
        save_image(data.cpu().data, 'real_image_{:02d}.png'.format(epoch), nrow=10)
        save_image(recon_x.cpu().data, 'recon_image_{:02d}.png'.format(epoch), nrow=10)
        save_image(decoded.cpu().data, 'sample_image_{:02d}.png'.format(epoch), nrow=10)
        if epoch % 50 == 0:
            torch.save(encoder.state_dict(), 'encoder_{:02d}.pth'.format(epoch))
            torch.save(decoder.state_dict(), 'decoder_{:02d}.pth'.format(epoch))
            torch.save(discriminator.state_dict(), 'discriminator_{:02d}.pth'.format(epoch))

    # add the learning rate adjust function
    def adjust_learning_rate_manual(optimizer, epoch):
        for param_group in optimizer.param_groups:
            if epoch == 30:
                param_group['lr'] /= 2.
            elif epoch == 50:
                param_group['lr'] /= 2.5
            elif epoch == 100:
                param_group['lr'] /= 2.

    print("=========> Pretrain encoder")
    for epoch in range(1, pretrain_epochs+1):
        encoder_pretrain(epoch)
    print("=========> Train models")
    for epoch in range(1, epochs+1):
        # do not need to use adjust_learning_rate in mnist
        # adjust_learning_rate_manual(optimizer_ae, epoch)
        # adjust_learning_rate_manual(optimizer_d, epoch)
        train(epoch)


if __name__ == "__main__":
    run()
