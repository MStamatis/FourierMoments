# FourierMoments
2D Fourier Moments Python Implementation

Ref: Implements `moments basis tomography` from
https://www.researchgate.net/publication/320567826_A_Web_Based_Scheme_for_Image_Tomography_Applications
in python

-> The input (i.e. image) must has the same width & height(x,y) and be 2D (black & white). If not, you should convert it.

from PIL import Image

import numpy as np

from numpy import *

im = array(Image.open(r"MyPNG.png"))

S_Out = FourierMoments(im,[180,180])

from matplotlib import pyplot as plt

plt.imshow(S_Out)

plt.show()


