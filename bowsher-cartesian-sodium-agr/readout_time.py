import numpy as np


def readout_time(k,
                 lookup=None,
                 eta=0.9830,
                 c1=0.54,
                 c2=0.46,
                 alpha_sw_tpi=18.95,
                 beta_sw_tpi=-0.5171,
                 t0_sw=0.0018):

    if lookup is not None:
        # lookup file: column 0 = time (ms), column 1 = k
        tmp = np.loadtxt(lookup)
        # np.interp needs increasing x -> sort by k
        # (also makes trajectories that run from the edge inwards usable,
        #  as long as k(t) is monotonic)
        order = np.argsort(tmp[:, 1], kind='stable')
        x = tmp[order, 1]
        y = tmp[order, 0] / 1000

        t = np.interp(k, x, y)

    else:
        # the point until the readout is linear
        m = 1126 * 0.16

        k_lin = t0_sw * m

        i1 = np.where(k <= k_lin)
        i2 = np.where(k > k_lin)

        t = np.zeros(k.shape)
        t[i1] = k[i1] / m
        t[i2] = t0_sw + ((c2 * (k[i2]**3) -
                          ((c1 / eta) * np.exp(-eta *
                                               (k[i2]**3))) - beta_sw_tpi) /
                         (3 * alpha_sw_tpi))

    return t

def max_k_from_lookup(lookup):
    """maximum k value contained in a k-space time lookup file"""
    return np.loadtxt(lookup)[:, 1].max()
 
 
def setup_readout_bins(recon_shape,
                       data_shape,
                       k_edge,
                       n_readout_bins,
                       readout_kwargs=None,
                       kedge_radius=None,
                       readout_duration_ms=None):
    """ Bin the (fft-ordered) recon k-space grid into shells of equal |k| and
    assign a readout time to every shell.
 
    Parameters
    ----------
    recon_shape : shape of the reconstruction grid, e.g. (256,256,256)
    data_shape : shape of the acquired data matrix, e.g. (128,128,128)
    k_edge : k value (in the units of the readout time model) at the
             edge of the acquired k-space
    n_readout_bins : number of k-space shells / readout time bins
    readout_kwargs : dict passed to readout_time (e.g. {'lookup': 'traj.ksp'})
    kedge_radius : radius (in recon grid units) of the acquired k-space edge
                   along each axis. Default: data_shape / 2
    readout_duration_ms : if given, all readout times are scaled by a common
                   factor such that the time at k_edge equals this value.
                   Valid when only the gradient strength (speed) differs
                   from the trajectory model, not its shape.
 
    Returns
    -------
    readout_inds : list of index tuples (one per bin)
    tr : 1d array, mean readout time of each bin in ms
    read_out_img : array of recon_shape, bin number (1..n) of every k-space
                   point, 0 for points outside the acquired k-space
    """
    if readout_kwargs is None:
        readout_kwargs = {}
 
    recon_shape = tuple(int(x) for x in recon_shape)
 
    if kedge_radius is None:
        kedge_radius = np.array(data_shape, dtype=float) / 2
    else:
        kedge_radius = np.broadcast_to(np.asarray(kedge_radius, dtype=float),
                                       (len(recon_shape), )).copy()
 
    # k-space coordinates (centered), rescaled such that k = k_edge
    # is at the edge of the acquired k-space along every axis
    k1d = [(np.arange(nn) - nn // 2 + 0.5) * (k_edge / r)
           for nn, r in zip(recon_shape, kedge_radius)]
    kk = np.meshgrid(*k1d, indexing='ij')
    abs_k = np.sqrt(sum(x**2 for x in kk))
    abs_k = np.fft.fftshift(abs_k)
 
    # readout times in ms
    t_read = 1000 * readout_time(abs_k, **readout_kwargs)
 
    if readout_duration_ms is not None:
        t_end = 1000 * readout_time(np.array([k_edge]), **readout_kwargs)[0]
        t_read *= readout_duration_ms / t_end
 
    k_1d = np.linspace(0, k_edge, n_readout_bins + 1)
 
    readout_inds = []
    tr = np.zeros(n_readout_bins)
    read_out_img = np.zeros(recon_shape, dtype=np.int32)
 
    for i in range(n_readout_bins):
        # half-open bins [k_start, k_end), last bin closed, so that every
        # k-space point belongs to exactly one bin
        if i < n_readout_bins - 1:
            sel = np.logical_and(abs_k >= k_1d[i], abs_k < k_1d[i + 1])
        else:
            sel = np.logical_and(abs_k >= k_1d[i], abs_k <= k_1d[i + 1])
 
        rinds = np.where(sel)
 
        if rinds[0].size == 0:
            raise ValueError(
                f'readout bin {i} is empty - reduce n_readout_bins')
 
        tr[i] = t_read[rinds].mean()
        readout_inds.append(rinds)
        read_out_img[rinds] = i + 1
 
    return readout_inds, tr, read_out_img
 


if __name__ == '__main__':
    import matplotlib.pyplot as py

    nk = 128
    T2s = 0.029
    T2f = 0.009
    osp = 1.8
    Km = osp * 0.8197

    rkwargs = {'lookup': 'kspace_times_4mm_g32.ksp'}

    #rkwargs = {'eta'          : 0.9830,
    #           'c1'           : 0.54,
    #           'c2'           : 0.46,
    #           'alpha_sw_tpi' : 18.95,
    #           'beta_sw_tpi'  : -0.5171,
    #           't0_sw'        : 0.0018}

    k = np.linspace(0, Km, nk)
    t = readout_time(k, **rkwargs)

    decay_env = 0.6 * np.exp(-t / T2f) + 0.4 * np.exp(-t / T2s)

    fig, ax = py.subplots(1, 4, figsize=(12, 3))
    ax[0].plot(1000 * t, k)
    ax[0].set_xlim((0, None))

    ax[1].plot(k, 1000 * t)
    #ax[2].axvline(rkwargs['t0_sw']*(1126*0.16), color = 'k', ls = ':')
    ax[2].plot(k, 1000 * t)
    ax[2].set_xlim((0, 0.5))
    ax[2].set_ylim((0, 3.5))

    ax[3].plot(k, decay_env)

    ax[0].set_xlabel('t (ms)')
    ax[0].set_ylabel('k')
    ax[1].set_xlabel('k')
    ax[1].set_ylabel('t( ms)')
    ax[2].set_xlabel('k (zoom)')
    ax[2].set_ylabel('t( ms)')
    ax[3].set_xlabel('k')
    ax[3].set_ylabel('decay env.')

    for axx in ax.flatten():
        axx.grid(ls=':')

    fig.suptitle(' '.join([x[0] + ':' + str(x[1]) for x in rkwargs.items()]),
                 fontsize='small')
    fig.tight_layout(pad=1.5)
    fig.show()
