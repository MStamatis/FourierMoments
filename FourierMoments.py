def FourierMoments(S, a):
  """
  Fourier Moments Basis
  Parameters
  ----------
  S : numpy array
      The image signal to be transformed.
  a : array
      number of moments (n,m)
  Returns
  -------
  S_out : numpy array
      The Moment Based transformed signal.
  References
  ---------
    .. [1] This algorithm implements `moments basis tomography` from
        https://www.researchgate.net/publication/320567826_A_Web_Based_Scheme_for_Image_Tomography_Applications
        in python
  """
  n = a[0]
  m = a[1]
  b=S.shape[0]
  bb=S.shape[1]

  B1=np.zeros((b,n))
  r=(n-1)/2
  for l in range(0,n):
      r2=(l-r-1)
      for k in range(1,b):
          z=-1j*pi*((k-1)*r2)/b
          B1[k,l]=np.exp(z)
  B1=(1/sqrt(b))*B1

  B2=np.zeros((b,m))
  r=(n-1)/2
  for l in range(0,m):
      r2=(l-r-1)
      for k in range(1,bb):
          z=-1j*pi*((k-1)*r2)/bb
          B2[k,l]=np.exp(z)
  B2=(1/np.sqrt(bb))*B2

  P = np.dot(np.dot(B1.T, np.double(S)), B2)

  S_out = np.dot(np.dot(np.linalg.pinv(B1.T), P), np.linalg.pinv(B2))

  return S_out
