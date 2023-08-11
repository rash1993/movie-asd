import matplotlib as mlp 
mlp.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pylab
import os
import sys
import scipy
from PIL import Image
import tempfile
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
# import aolib.img as im, aolib.util as ut
def to_pil(im):
  #print im.dtype
  return Image.fromarray(np.uint8(im))

def from_pil(pil):
  #print pil
  return np.array(pil)

def fail(s = ''): raise RuntimeError(s)

def resize(im, scale, order = 3, hires = False):
  if hires == 'auto':
    hires = (im.dtype == np.uint8)

  if np.ndim(scale) == 0:
    new_scale = [scale, scale]
  # interpret scale as dimensions; convert integer size to a fractional scale
  elif ((scale[0] is None) or type(scale[0]) == type(0)) \
           and ((scale[1] is None) or type(scale[1]) == type(0)) \
           and (not (scale[0] is None and scale[1] is None)):
    # if the size of only one dimension is provided, scale the other to maintain the right aspect ratio
    if scale[0] is None:
      dims = (int(float(im.shape[0])/im.shape[1]*scale[1]),  scale[1])
    elif scale[1] is None:
      dims = (scale[0], int(float(im.shape[1])/im.shape[0]*scale[0]))
    else:
      dims = scale[:2]

    new_scale = [float(dims[0] + 0.4)/im.shape[0], float(dims[1] + 0.4)/im.shape[1]]
    # a test to make sure we set the floating point scale correctly
    result_dims = [int(new_scale[0]*im.shape[0]), int(new_scale[1]*im.shape[1])]
    assert tuple(result_dims) == tuple(dims)
  elif type(scale[0]) == type(0.) and type(scale[1]) == type(0.):
    new_scale = scale
    #new_scale = scale[1], scale[0]
  else:
    raise RuntimeError("don't know how to interpret scale: %s" % (scale,))
    # want new scale' to be such that
    # int(scale'[0]*im.shape[0]) = scale[0], etc. (that's how zoom computes the new shape)
    # todo: any more numerical issues?
    #print 'scale before', im.shape, scale
    # print 'scale after', scale
    # print 'new image size', [int(scale[0]*im.shape[0]),int(scale[1]*im.shape[1])]
  #scale_param = new_scale if im.ndim == 2 else (new_scale[0], new_scale[1], 1)
  scale_param = new_scale if im.ndim == 2 else (new_scale[0], new_scale[1], 1)

  if hires:
    #sz = map(int, (scale_param*im.shape[1], scale_param*im.shape[0]))
    sz = map(int, (scale_param[1]*im.shape[1], scale_param[0]*im.shape[0]))
    return from_pil(to_pil(im).resize(sz, Image.ANTIALIAS))
  else:
    res = scipy.ndimage.zoom(im, scale_param, order = order)
    # verify that zoom() returned an image of the desired size
    if (np.ndim(scale) != 0) and type(scale[0]) == type(0) and type(scale[1]) == type(0):
      assert res.shape[:2] == (scale[0], scale[1])
    return res

def clip_rescale(x, lo = None, hi = None):
  if lo is None:
    lo = np.min(x)
  if hi is None:
    hi = np.max(x)
  return np.clip((x - lo)/(hi - lo), 0., 1.)

def apply_cmap(im, cmap = pylab.cm.jet, lo = None, hi = None):
  return cmap(clip_rescale(im, lo, hi).flatten()).reshape(im.shape[:2] + (-1,))[:, :, :3]

def cmap_im(cmap, im, lo = None, hi = None):
  return np.uint8(255*apply_cmap(im, cmap, lo, hi))

def save(img_fname, a):
  if img_fname.endswith('jpg'):
    return Image.fromarray(np.uint8(a)).save(img_fname, quality = 100)
  else:
    #return Image.fromarray(np.uint8(a)).save(img_fname)
    return Image.fromarray(np.uint8(a)).save(img_fname, quality = 100)

def make_file(fname, contents = ''):
  f = open(fname, 'w')
  f.write(contents)
  f.close()

def write_lines(fname, lines):
  assert type(lines) != type('')
  f = open(fname, 'w')
  for line in lines:
    f.write(line)
    f.write("\n")
  f.close()

def make_temp(ext, contents = None, dir = None):
  fd, fname = tempfile.mkstemp(ext, prefix = 'ao_', dir = '/home/azureuser/cloudfiles/code/temp')
  os.close(fd)
  if contents is not None:
    make_file(fname, contents)
  return os.path.abspath(fname)

class temp_file:
  def __init__(self, ext, fname_only = False, delete_on_exit = True):
    self.fname = make_temp(ext)
    self.delete_on_exit = delete_on_exit
    if fname_only:
      os.remove(self.fname)

  def __enter__(self):
    return self.fname

  def __exit__(self, type, value, tb):
    if self.delete_on_exit and os.path.exists(self.fname):
      os.remove(self.fname)

class temp_files:
  def __init__(self, ext, count, path = None, fname_only = False, delete_on_exit = True):
    self.fnames = [make_temp(ext, dir = path) for i in range(count)]
    self.delete_on_exit = delete_on_exit
    if fname_only:
      for fname in self.fnames:
        os.remove(fname)

  def __enter__(self):
    return self.fnames

  def __exit__(self, type, value, tb):
    if self.delete_on_exit:
      for fname in self.fnames:
        if os.path.exists(fname):
          os.remove(fname)

def sys_check(*args):
  cmd = ' '.join(args)
  print (cmd)
  if 0 != os.system(cmd):
    fail('Command failed! %s' % cmd)
  return 0

def make_video(im_fnames, fps, out_fname, sound_fname = None, keep_aud_file = False, flags = ''):
  # if type(sound_fname) != type(''):
  #   tmp_wav = make_temp('.wav')
  #   sound_fname.save(tmp_wav)
  #   sound_fname = tmp_wav
  # else:
  if sound_fname is  not None:
      tmp_wav = sound_fname
  else:
      tmp_wav = None

  write_ims = (type(im_fnames[0]) != type(''))
  num_ims = len(im_fnames) if write_ims else 0
  with temp_file('.txt') as input_file, temp_files('.ppm', num_ims) as tmp_ims:
    if write_ims:
      for fname, x in zip(tmp_ims, im_fnames):
        save(fname, x)
      im_fnames = tmp_ims

    write_lines(input_file, ['file %s' % fname for fname in im_fnames])
    sound_flags_in = ('-i "%s"' % sound_fname) if sound_fname is not None else ''
    sound_flags_out =  '-acodec aac' if sound_fname is not None else ''
    #os.system('echo input file; cat %s' % input_file)
    sys_check('ffmpeg -y -nostdin %s -r %f -loglevel error -safe 0 -f concat -i "%s" -pix_fmt yuv420p -vcodec h264 -strict -2 -y %s %s "%s"' \
              % (sound_flags_in, fps, input_file, sound_flags_out, flags, out_fname))

  print (tmp_wav)
  if tmp_wav is not None:
    if keep_aud_file:
        return
    else:
        os.remove(tmp_wav)

def accumulate_cams(metadata_file, cams):
  print('accumulating cams')
  # cams = (cams - np.min(cams))/(np.max(cams) - np.min(cams))
  max_val = np.percentile(cams, 97)
  lo_frac = 0.5
  max_prob = 0.35
  metadata = np.load(metadata_file)
  done_counter = 0
  cams_out = {}
  total_frames = np.sum(metadata[:,0])
  for i, scene in enumerate(metadata):
    sf = done_counter + scene[1]
    ef = done_counter + scene[2]
    done_counter = done_counter + scene[0]
    out = []
    for frame_no in range(sf, ef):
      lo = lo_frac*max_val
      hi = max_val + 0.001
      f = cams.shape[0]*float(frame_no) / total_frames
      l = int(f)
      r = min(1 + l, cams.shape[0]-1)
      p = f - l
      frame_cam = (1 -p)*cams[l] + p*cams[r]
      frame_cam = resize(frame_cam, [180, 360], 1)
      vis = cmap_im(pylab.cm.jet, frame_cam, lo=lo, hi=hi)
      out.append(frame_cam)
    cams_out[i+1] = out
  return cams_out


def heatmap(frames, cam, lo_frac = 0.5, adapt = True, max_val = 35):
    """ Set heatmap threshold adaptively, to deal with large variation in possible input videos. """
    # print ('got into right one')
    frames = np.asarray(frames)
    max_prob = 0.35
    if adapt:
        max_val = np.percentile(cam, 97)

    same = np.max(cam) - np.min(cam) <= 0.001
    if same:
        print ('returning if cams are same')
        return frames

    outs = []
    scaled_cams = []
    fcams =[]
    # vis_lores = []
    for i in range(frames.shape[0]):
        lo = lo_frac * max_val
        hi = max_val + 0.001
        im = frames[i]
        f = cam.shape[0] * float(i) / frames.shape[0]
        l = int(f)
        r = min(1 + l, cam.shape[0]-1)
        p = f - l
        frame_cam_lores = ((1-p) * cam[l]) + (p * cam[r])
        frame_cam = resize(frame_cam_lores, im.shape[:2], 1)
        fcams.append(frame_cam)
        #vis = ut.cmap_im(pylab.cm.hot, np.minimum(frame_cam, hi), lo = lo, hi = hi)
        vis = cmap_im(pylab.cm.jet, frame_cam, lo = lo, hi = hi)
        # vislo = ut.cmap_im(pylab.cm.jet, frame_cam, lo = lo, hi = hi)
        #p = np.clip((frame_cam - lo)/float(hi - lo), 0, 1.)
        p = np.clip((frame_cam - lo)/float(hi - lo), 0, max_prob)
        p = p[..., None]
        im = np.array(im, 'd')
        vis = np.array(vis, 'd')
        # vislo = np.array(vis, 'd')
        outs.append(np.uint8(im*(1-p) + vis*p))
        scaled_cams.append(vis)
        # vis_lores.append(vislo)
    print (np.array(outs).shape)
    return np.array(outs), np.array(scaled_cams), np.array(fcams)


def impose_labels(name, labs):
    audio_name = name[:-4] + '_audio.aac'
    os.system('ffmpeg -nostdin -i %s -vn -acodec copy %s' %(name, audio_name))
    flag = True
    frames = []
    vid = cv2.VideoCapture(name)
    FPS = vid.get(5)
    while(flag):
        flag, img = vid.read()
        if flag:
            frames.append(img)
    frames = np.array(frames)
    y_shape = frames.shape[1]
    x_shape = frames.shape[2]
    center = (x_shape - 50 , 50)
    radius = 50
    frame_labs = np.zeros(frames.shape[0]).astype(np.bool)
    # fps = vid.get(7)/10.0
    fps = FPS
    for i in range(labs.shape[0]):
        if labs[i]:
            st_frame = int(np.floor(fps*i))
            ed_frame = int(st_frame + np.floor(fps))
            frame_labs[st_frame:ed_frame] = True
    for i in range(frames.shape[0]):
        if frame_labs[i]:
            cv2.circle(frames[i], center, radius, (0, 225, 0), -1)
        else:
            cv2.circle(frames[i], center, radius, (0, 0, 255), -1)
    for i in range(frames.shape[0]):
        frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
    make_video(frames, FPS, name[:-4]+'_impose.mp4', sound_fname=audio_name)


def _proposal_per_frames(f, p, m):
  z_ =  zip(p, m)
  # out = []

  for p_, m_ in z_:
    # print()
    if m_>0:
      for f_ in f:
        # print ('drawn')
        cv2.rectangle(f_, (p_[0], p_[1]), (p_[2], p_[3]), (255, 0, 0), 2)
  return f


def proposals_on_frames(frames, mask, proposals):
  # input is every sample (of N sec)
  proposals[:,:,0] = proposals[:,:,0]*360.0
  proposals[:,:,1] = proposals[:,:,1]*180.0
  proposals[:,:,2] = proposals[:,:,2]*360.0
  proposals[:,:,3] = proposals[:,:,3]*180.0
  N = len(frames)
  P = proposals.shape[-2]
  proposals = np.reshape(proposals, [N, 6, P, 4])
  mask = np.reshape(mask[:,:6*P], [N, 6, P])
  z = zip(frames, proposals, mask)
  frames_out = []
  for seg_frames, seg_prop, seg_mask in z:
    for j in range(len(seg_prop)):
      frames_out.extend(_proposal_per_frames(seg_frames[j*4: (j+1)*4], seg_prop[j], seg_mask[j]))
  return frames_out

def plot_proposals(frames, mask, proposals):
  z = zip(frames, mask, proposals)
  out = []
  for f, m, p in z:
    out.extend(proposals_on_frames(f, m, p))
  return out

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax