import argparse
import model

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nr_lr', type=float, default=2e-5)
parser.add_argument('--gan_lr', type=float, default=2e-4)
parser.add_argument('--dom_loss', type=float, default=100)
parser.add_argument('--z_size', type=float, default=200)
parser.add_argument('--bias', type=bool, default=False)
parser.add_argument('--dropout_rate', type=float, default=0.25)
parser.add_argument('--is_grayscale', type=bool, default=True)

args = parser.parse_args()

generator = model.Generator(args)
renderer = model.RenderNet(args)
discriminator = model.Discriminator(args)
