import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.z_size = args.z_size
        self.cube_len = 64
        self.bias = args.bias

        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(self.z_size, self.cube_len * 8, kernel_size=4, stride=1, bias=self.bias,
                               padding=(0, 0, 0)),
            nn.BatchNorm3d(self.cube_len * 8),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(self.cube_len * 8, self.cube_len * 4, kernel_size=4, stride=2, bias=self.bias,
                               padding=(1, 1, 1)),
            nn.BatchNorm3d(self.cube_len * 4),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(self.cube_len * 4, self.cube_len * 2, kernel_size=4, stride=2, bias=self.bias,
                               padding=(1, 1, 1)),
            nn.BatchNorm3d(self.cube_len * 2),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(self.cube_len * 2, self.cube_len, kernel_size=4, stride=2, bias=self.bias,
                               padding=(1, 1, 1)),
            nn.BatchNorm3d(self.cube_len),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(self.cube_len, 1, kernel_size=4, stride=2, bias=self.bias,
                               padding=(1, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, self.z_size, 1, 1, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out


class RenderNet(nn.Module):

    def __init__(self, args):
        self.bias = args.bias
        self.prob = args.dropout_rate
        self.is_grayscale = args.is_grayscale
        super(RenderNet, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=5, stride=(2, 2, 2), padding=2, bias=self.bias),
            nn.PReLU(),
            nn.Dropout3d(self.prob)
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, stride=(2, 1, 1), padding=1, bias=self.bias),
            nn.PReLU(),
            nn.Dropout3d(self.prob)
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=(1, 1, 1), padding=1, bias=self.bias),
            nn.PReLU(),
            nn.Dropout3d(self.prob)
        )

        self.res3d1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.PReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=self.bias),
        )
        self.res3d2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.PReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=self.bias),
        )

        self.proj = nn.Sequential(
            nn.Conv2d(32 * 16, 32 * 16, kernel_size=1, stride=1, bias=self.bias),
            nn.PReLU()
        )

        self.res2d1 = nn.Sequential(
            nn.Conv2d(32 * 16, 32 * 16, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.PReLU(),
            nn.Conv2d(32 * 16, 32 * 16, kernel_size=3, stride=1, padding=1, bias=self.bias),
        )
        self.res2d2 = nn.Sequential(
            nn.Conv2d(32 * 16, 32 * 16, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.PReLU(),
            nn.Conv2d(32 * 16, 32 * 16, kernel_size=3, stride=1, padding=1, bias=self.bias),
        )

        self.enc4 = nn.Sequential(
            nn.ConstantPad2d((1, 2, 1, 2), 0),
            nn.Conv2d(32 * 16, 32 * 16, kernel_size=4, stride=1, padding=0, bias=self.bias),
            nn.PReLU(),
            nn.Dropout2d(self.prob)
        )

        self.res2d3 = nn.Sequential(
            nn.Conv2d(32 * 16, 32 * 16, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.PReLU(),
            nn.Conv2d(32 * 16, 32 * 16, kernel_size=3, stride=1, padding=1, bias=self.bias),
        )
        self.res2d4 = nn.Sequential(
            nn.Conv2d(32 * 16, 32 * 16, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.PReLU(),
            nn.Conv2d(32 * 16, 32 * 16, kernel_size=3, stride=1, padding=1, bias=self.bias),
        )

        self.enc5 = nn.Sequential(
            nn.ConstantPad2d((1, 2, 1, 2), 0),
            nn.Conv2d(32 * 16, 32 * 8, kernel_size=4, stride=1, padding=0, bias=self.bias),
            nn.PReLU(),
            nn.Dropout2d(self.prob)
        )
        self.enc6 = nn.Sequential(
            nn.ConvTranspose2d(32 * 8, 32 * 4, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.PReLU(),
            nn.Dropout2d(self.prob)
        )
        self.enc7 = nn.Sequential(
            nn.ConstantPad2d((0, 1, 0, 1), 0),
            nn.ConvTranspose2d(32 * 4, 32 * 4, kernel_size=4, stride=1, padding=2, bias=self.bias),
            nn.PReLU(),
            nn.Dropout2d(self.prob)
        )
        self.enc8 = nn.Sequential(
            nn.ConvTranspose2d(32 * 4, 32 * 2, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.PReLU(),
            nn.Dropout2d(self.prob)
        )
        self.enc9 = nn.Sequential(
            nn.ConvTranspose2d(32 * 2, 32, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.PReLU(),
            nn.Dropout2d(self.prob)
        )
        self.enc10 = nn.Sequential(
            nn.ConstantPad2d((0, 1, 0, 1), 0),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=1, padding=2, bias=self.bias),
            nn.PReLU(),
            nn.Dropout2d(self.prob)
        )
        if self.is_grayscale:
            self.enc11 = nn.Sequential(
                nn.ConstantPad2d((0, 1, 0, 1), 0),
                nn.ConvTranspose2d(16, 1, kernel_size=4, stride=1, padding=2, bias=self.bias),
                nn.Sigmoid()
            )
        else:
            self.enc11 = nn.Sequential(
                nn.ConstantPad2d((0, 1, 0, 1), 0),
                nn.ConvTranspose2d(16, 3, kernel_size=4, stride=1, padding=2, bias=self.bias),
                nn.Sigmoid()
            )

    def forward(self, x):
        out = x
        out = self.enc1(out)
        out = self.enc2(out)
        out = self.enc3(out)

        out = torch.add(self.res3d1(out), out)
        out = torch.add(self.res3d2(out), out)

        # (input, channels, depth, width, height)
        out = out.view(-1, 32 * 16, 32, 32)
        out = self.proj(out)
        out = torch.add(self.res2d1(out), out)
        out = torch.add(self.res2d2(out), out)
        out = self.enc4(out)
        out = torch.add(self.res2d3(out), out)
        out = torch.add(self.res2d4(out), out)

        out = self.enc5(out)
        out = self.enc6(out)
        out = self.enc7(out)
        out = self.enc8(out)
        out = self.enc9(out)
        out = self.enc10(out)
        out = self.enc11(out)

        return out


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.channels = 1 if args.is_grayscale else 3
        self.bias = args.bias
        self.layer1 = nn.Sequential(
            # input is (nc) x 256 x 256
            SpectralNorm(nn.Conv2d(self.channels, 64, 4, 2, 1, bias=self.bias)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            # state size. (ndf) x 128 x 128
            SpectralNorm(nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=self.bias)),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            # state size. (ndf*2) x 64 x 64
            SpectralNorm(nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=self.bias)),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            # state size. (ndf*4) x 32 x 32
            SpectralNorm(nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=self.bias)),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5 = nn.Sequential(
            # state size. (ndf*8) x 16 x 16
            SpectralNorm(nn.Conv2d(64 * 8, 64 * 16, 4, 2, 1, bias=self.bias)),
            nn.BatchNorm2d(64 * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer6 = nn.Sequential(
            # state size. (ndf*16) x 8 x 8
            SpectralNorm(nn.Conv2d(64 * 16, 64 * 32, 4, 2, 1, bias=self.bias)),
            nn.BatchNorm2d(64 * 32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer7 = nn.Sequential(
            # state size. (ndf*32) x 4 x 4
            SpectralNorm(nn.Conv2d(64 * 32, 1, 4, 1, 0, bias=self.bias)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = x
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        return out
