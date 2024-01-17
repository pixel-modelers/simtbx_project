"""
This code provides methods to calculate the qscore metric for map-model validation,
as developed by Pintile et al. 

As with the original implementation, multiple calculation options are provided. 
The fastest (default) is to pass an mmtbx map-model-manager (mmm)
to the the function qscore_np, qscore_np(mmm)
"""

from __future__ import division
import math
import sys
import numpy as np
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count

from cctbx.array_family import flex
from scitbx_array_family_flex_ext import bool as flex_bool

from .qscore_utils import (
    trilinear_interpolation,
    rowwise_corrcoef
    sphere_points_np,
    sphere_points_flex,
    cdist_flex,
    query_ball_point_flex,
    query_atom_neighbors,
)
from .flex_utils import optimized_nd_to_1d_indices, nd_to_1d_indices, flex_std

try:
  from tqdm import tqdm
except ImportError:
  from .qscore_utils import DummyTQDM as tqdm


def radial_shell_worker_v1_np(args):
  """
  Calulate qscore for a single radial shell using version 1 (serial probe allocation) and numpy
  """
  (
      i,
      atoms_xyz,
      n_probes,
      radius_shell,
      tree,
      rtol,
      selection,
      n_probes_target,
  ) = args

  #
  # manage selection input
  if selection is None:
    selection = np.arange(len(atoms_xyz))
  else:
    assert selection.dtype == bool
  # do selection
  atoms_xyz_sel = atoms_xyz[selection]
  # print("sel_shape",atoms_xyz_sel.shape)
  n_atoms = atoms_xyz_sel.shape[0]

  if radius_shell == 0:
    radius_shell = 1e-9  # zero causes crash
  numPts = n_probes_target
  RAD = radius_shell
  outRAD = rtol
  all_pts = []  # list of probe arrays for each atom
  probe_xyz_r = np.full((n_atoms, n_probes_target, 3), -1.0)
  # print(atoms_xyz_sel)
  # print("n_atoms",n_atoms)
  for atom_i in range(n_atoms):
    coord = atoms_xyz_sel[atom_i]
    # print("atom_i",atom_i)
    # print("coord:",coord)
    pts = []

    # try to get at least [numPts] points at [RAD] distance
    # from the atom, that are not closer to other atoms
    for i in range(0, 50):
      # if we find the necessary number of probes in the first iteration, then i will never go to 1
      # points on a sphere at radius RAD...
      n_pts_to_grab = (
          numPts + i * 2
      )  # progressively more points are grabbed  with each failed iter
      # print("n_to_grab:",n_pts_to_grab)

      outPts = sphere_points_np(
          coord[None, :], RAD, n_pts_to_grab
      )  # get the points
      outPts = outPts.reshape(-1, 3)
      at_pts, at_pts_i = [None] * len(outPts), 0
      # print("probe candidates")

      for pt_i, pt in enumerate(
          outPts
      ):  # identify which ones to keep, progressively grow pts list
        # print(f"\t{pt[0]},{pt[1]},{pt[2]}")
        # query kdtree to find probe-atom interactions
        counts = tree.query_ball_point(
            pt[None, :], RAD * outRAD, return_length=True
        )

        # each value in counts is the number of atoms within radius+tol of each probe
        count = counts.flatten()[0]
        ptsNear = count

        if ptsNear == 0:
          at_pts[at_pts_i] = pt
          at_pts_i += 1
        # if at_pts_i >= numPts:
        #     break

      if (
          at_pts_i >= numPts
      ):  # if we have enough points, take all the "good" points from this iter
        pts.extend(at_pts[0:at_pts_i])
        break
    # assert len(pts)>0, "Zero probes were found "
    pts = np.array(pts)  # should be shape (n_probes,3)
    all_pts.append(pts)

  # prepare output
  n_atoms = len(atoms_xyz)
  for i, r in enumerate(all_pts):
    if r.ndim == 2 and len(r) > 0:
      probe_xyz_r[i, :n_probes, :] = r[:n_probes_target, :]

  keep_sel = probe_xyz_r != -1.0
  keep_sel = np.mean(keep_sel, axis=-1, keepdims=True)
  keep_sel = np.squeeze(keep_sel, axis=-1)

  return probe_xyz_r, keep_sel.astype(bool)


def radial_shell_worker_v2_np(args):
  """
  Calulate qscore for a single radial shell using version 2 (parallel probe allocation) and numpy
  """
  # unpack args
  i, atoms_xyz, n_probes, radius_shell, tree, rtol, selection = args

  # manage selection input
  if selection is None:
    selection = np.arange(len(atoms_xyz))
  else:
    assert selection.dtype == bool

  # do selection
  atoms_xyz_sel = atoms_xyz[selection]
  n_atoms = atoms_xyz_sel.shape[0]

  # get probe coordinates
  probe_xyz = sphere_points_np(atoms_xyz_sel, radius_shell, n_probes)

  counts = tree.query_ball_point(
      probe_xyz, radius_shell * rtol, return_length=True
  )  # atom counts for each probe, for probes in shape (n_atoms,n_probes)
  probe_mask = (
      counts == 0
  )  # keep probes with 0 nearby atoms. The rtol ensures self is not counted

  return probe_xyz, probe_mask


def radial_shell_mp_np(
  
    model,
    n_probes=32,
    radii=np.linspace(0.1, 2, 12),
    rtol=0.9,
    num_processes=cpu_count(),
    selection=None,
    version=2,
    log=sys.stdout,
  ):
  """
  Generate probes for a model file using numpy
  """
  assert version in [1, 2], "Version must be 1 or 2"

  atoms_xyz = model.get_sites_cart().as_numpy_array()
  tree = KDTree(atoms_xyz)

  if version == 1:
    worker_func = radial_shell_worker_v1_np
    n_probes_target = n_probes
    # Create argument tuples for each chunk
    args = [
        (
            i,
            atoms_xyz,
            n_probes,
            radius_shell,
            tree,
            rtol,
            selection,
            n_probes_target,
        )
        for i, radius_shell in enumerate(radii)
    ]
  else:
    worker_func = radial_shell_worker_v2_np
    # Create argument tuples for each chunk
    args = [
        (i, atoms_xyz, n_probes, radius_shell, tree, rtol, selection)
        for i, radius_shell in enumerate(radii)
    ]

  # Create a pool of worker processes
  if num_processes > 1:
    with Pool(num_processes) as p:
      # Use the pool to run the trilinear_interpolation_worker function in parallel
      results = list(
          tqdm(p.imap(worker_func, args), total=len(args), file=log)
      )
  else:
    results = []
    for arg in tqdm(args, file=log):
      # for arg in args:
      result = worker_func(arg)
      results.append(result)

  probe_xyz_all = [result[0] for result in results]
  probe_mask_all = [result[1] for result in results]

  # debug
  # return probe_xyz_all, probe_mask_all

  # stack numpy
  probe_xyz = np.stack(probe_xyz_all)
  probe_mask = np.stack(probe_mask_all)

  return probe_xyz, probe_mask


def qscore_np(
    mmm,
    selection=None,
    n_probes=32,
    shells=np.array(
        [
            0.1,
            0.27272727,
            0.44545455,
            0.61818182,
            0.79090909,
            0.96363636,
            1.13636364,
            1.30909091,
            1.48181818,
            1.65454545,
            1.82727273,
            2.0,
        ]
    ),
    version=2,
    nproc=cpu_count(),
    log=sys.stdout,
  ):
  """
  Calculate the qscore metric per-atom from an mmtbx map-model-manager, using numpy
  """
  model = mmm.model()
  mm = mmm.map_manager()
  volume = mm.map_data().as_numpy_array()
  radii = shells
  voxel_size = mm.pixel_sizes()

  probe_xyz, probe_mask = radial_shell_mp_np(
      model,
      n_probes=n_probes,
      num_processes=nproc,
      selection=selection,
      version=version,
      radii=radii,
      log=log,
  )
  # return probe_xyz,probe_mask
  # after the probe generation, versions 1 and 2 are the same

  # infer params from shape
  n_shells, n_atoms, n_probes, _ = probe_xyz.shape

  # flatten
  probe_xyz_flat = probe_xyz.reshape((n_atoms * n_shells * n_probes, 3))
  probe_mask_flat = probe_mask.reshape(-1)  # (n_shells*n_atoms*n_probes,)

  # select mask=True probes
  masked_probe_xyz_flat = probe_xyz_flat[probe_mask_flat]

  # interpolate
  masked_density = trilinear_interpolation(
      volume, masked_probe_xyz_flat, voxel_size=voxel_size
  )

  # reshape interpolated values to (n_shells,n_atoms, n_probes)

  d_vals = np.zeros((n_shells, n_atoms, n_probes))
  d_vals[probe_mask] = masked_density

  # reshape to (M,N*L) for rowwise correlation

  d_vals_2d = d_vals.transpose(1, 0, 2).reshape(d_vals.shape[1], -1)

  # create the reference data

  M = volume
  maxD = min(M.mean() + M.std() * 10, M.max())
  minD = max(M.mean() - M.std() * 1, M.min())
  A = maxD - minD
  B = minD
  u = 0
  sigma = 0.6
  x = np.array(radii)
  y = A * np.exp(-0.5 * ((x - u) / sigma) ** 2) + B

  # Stack and reshape data for correlation calc

  # stack the reference to shape (n_shells,n_atoms,n_probes)
  g_vals = np.repeat(y[:, None], n_probes, axis=1)
  g_vals = np.expand_dims(g_vals, 1)
  g_vals = np.tile(g_vals, (n_atoms, 1))

  # reshape
  g_vals_2d = g_vals.transpose(1, 0, 2).reshape(g_vals.shape[1], -1)
  d_vals_2d = d_vals.transpose(1, 0, 2).reshape(d_vals.shape[1], -1)
  mask_2d = probe_mask.transpose(1, 0, 2).reshape(probe_mask.shape[1], -1)

  # # CALCULATE Q

  # # numpy
  q = rowwise_corrcoef(g_vals_2d, d_vals_2d, mask=mask_2d)

  return q


##############################################################################
# Functions that use only flex, no numpy
##############################################################################


def radial_shell_worker_v2_flex(args):
  """
  Calulate qscore for a single radial shell using version 2 (parallel probe allocation) and flex only
  """
  # unpack args
  i, atoms_xyz, n_probes, radius_shell, rtol, tree, selection = args

  # manage selection input
  if selection is None:
    selection = flex.size_t_range(len(atoms_xyz))
  else:
    assert isinstance(selection, flex_bool)

  # do selection
  n_atoms = selection.count(True)
  atoms_xyz_sel = atoms_xyz.select(selection)

  # get probe coordinates
  probe_xyz = sphere_points_flex(atoms_xyz_sel, radius_shell, n_probes)

  # query to find the number of atoms within the clash range of each probe
  counts = query_ball_point_flex(
      tree, atoms_xyz, probe_xyz, r=radius_shell * rtol
  )
  probe_mask = counts == 0
  return probe_xyz, probe_mask


def radial_shell_v2_mp_flex(
    model,
    n_probes=32,
    radii=np.linspace(0.1, 2, 12),
    rtol=0.9,
    num_processes=cpu_count(),
    selection=None,
    version=2,
    log=sys.stdout,
  ):
  """
  Generate probes for a model file using flex only
  """
  assert version in [1, 2], "Version must be 1 or 2"
  if version == 1:
    assert False
  else:
    worker_func = radial_shell_worker_v2_flex

  # get a "tree", which is just a dictionary of index:local neighbor indices
  tree, _ = query_atom_neighbors(model, radius=3.5)
  atoms_xyz = model.get_sites_cart()

  # i,atoms_xyz,n_probes,radius_shell,rtol, selection= args
  # Create argument tuples for each chunk

  args = [
      (i, atoms_xyz, n_probes, radius_shell, rtol, tree, selection)
      for i, radius_shell in enumerate(radii)
  ]

  # Create a pool of worker processes
  if num_processes > 1:
    with Pool(num_processes) as p:
      # Use the pool to run the trilinear_interpolation_worker function in parallel
      results = list(
          tqdm(p.imap(worker_func, args), total=len(args), file=log)
      )
  else:
    results = []
    for arg in tqdm(args, file=log):
      # for arg in args:
      result = worker_func(arg)
      results.append(result)

  # stack the results from each shell into single arrays
  probe_xyz_all = [result[0] for result in results]
  probe_mask_all = [result[1] for result in results]

  # # debug
  # return probe_xyz_all, probe_mask_all,tree

  n_atoms = probe_xyz_all[0].focus()[0]
  n_shells = len(probe_mask_all)
  out_shape = (n_shells, n_atoms, n_probes, 3)
  out_size = math.prod(out_shape)
  shell_size = math.prod(out_shape[1:])
  out_probes = flex.double(out_size, -1.0)
  out_mask = flex.bool(n_atoms * n_shells * n_probes, False)

  for i, p in enumerate(probe_xyz_all):
    start = i * shell_size
    stop = start + shell_size
    out_probes = out_probes.set_selected(
        flex.size_t_range(start, stop), p.as_1d()
    )
  out_probes.reshape(flex.grid(*out_shape))

  for i, k in enumerate(probe_mask_all):
    start = i * (n_atoms * n_probes)
    stop = start + (n_atoms * n_probes)
    out_mask = out_mask.set_selected(
        flex.size_t_range(start, stop), k.as_1d()
    )
  out_mask.reshape(flex.grid(n_shells, n_atoms, n_probes))

  return out_probes, out_mask


def qscore_flex(
    mmm,
    selection=None,
    n_probes=32,
    shells=[
        0.1,
        0.27272727,
        0.44545455,
        0.61818182,
        0.79090909,
        0.96363636,
        1.13636364,
        1.30909091,
        1.48181818,
        1.65454545,
        1.82727273,
        2.0,
    ],
    version=2,
    nproc=cpu_count(),
  ):
  """
  Calculate the qscore metric per-atom from an mmtbx map-model-manager, using flex only
  """
  model = mmm.model()
  mm = mmm.map_manager()
  radii = shells
  volume = mm.map_data()
  voxel_size = mm.pixel_sizes()

  probe_xyz, probe_mask = radial_shell_v2_mp_flex(
      model,
      n_probes=n_probes,
      num_processes=nproc,
      selection=selection,
      version=version,
      radii=radii,
  )

  # aliases
  probe_xyz_cctbx = probe_xyz
  probe_mask_cctbx = probe_mask

  # infer params from shape
  n_shells, n_atoms, n_probes, _ = probe_xyz.focus()

  # APPLY MASK BEFORE INTERPOLATION

  probe_mask_cctbx_fullflat = []

  for val in probe_mask_cctbx:
    for _ in range(3):  # since A has an additional dimension of size 3
      probe_mask_cctbx_fullflat.append(val)

  mask = flex.bool(probe_mask_cctbx_fullflat)
  # indices = flex.int([i for i in range(1, keep_mask_cctbx.size() + 1) for _ in range(3)])
  sel = probe_xyz_cctbx.select(mask)
  # sel_indices = indices.select(mask)
  masked_probe_xyz_flat_cctbx = flex.vec3_double(sel)

  # INTERPOLATE

  masked_density_cctbx = mm.density_at_sites_cart(
      masked_probe_xyz_flat_cctbx
  )

  # reshape interpolated values to (n_shells,n_atoms, n_probes)

  probe_mask_cctbx.reshape(flex.grid(n_shells * n_atoms * n_probes))
  d_vals_cctbx = flex.double(probe_mask_cctbx.size(), 0.0)
  d_vals_cctbx = d_vals_cctbx.set_selected(
      probe_mask_cctbx, masked_density_cctbx
  )
  d_vals_cctbx.reshape(flex.grid(n_shells, n_atoms, n_probes))

  # reshape to (M,N*L) for rowwise correlation

  def custom_reshape_indices(flex_array):
    N, M, L = flex_array.focus()
    result = flex.double(flex.grid(M, N * L))

    for i in range(N):
      for j in range(M):
        for k in range(L):
          # Calculate the original flat index
          old_index = i * M * L + j * L + k
          # Calculate the new flat index after transpose and reshape
          new_index = j * N * L + i * L + k
          result[new_index] = flex_array[old_index]

    return result

  d_vals_2d_cctbx = custom_reshape_indices(d_vals_cctbx)

  # create the reference data
  M = volume
  M_std = flex_std(M)
  M_mean = flex.mean(M)
  maxD_cctbx = min(M_mean + M_std * 10, flex.max(M))
  minD_cctbx = max(M_mean - M_std * 1, flex.min(M))
  A_cctbx = maxD_cctbx - minD_cctbx
  B_cctbx = minD_cctbx
  u = 0
  sigma = 0.6
  x = flex.double(radii)
  y_cctbx = (
      A_cctbx * flex.exp(-0.5 * ((flex.double(x) - u) / sigma) ** 2)
      + B_cctbx
  )

  # Stack and reshape data for correlation calc

  # 1. Repeat y for n_probes (equivalent to np.repeat)
  g_vals_cctbx = [[val] * n_probes for val in y_cctbx]

  # 2. Add a new dimension (equivalent to np.expand_dims)
  g_vals_expanded = [[item] for item in g_vals_cctbx]

  # 3. Tile for each atom (equivalent to np.tile)
  g_vals_tiled = []
  for item in g_vals_expanded:
    g_vals_tiled.append(item * n_atoms)

  g_vals_cctbx = flex.double(np.array(g_vals_tiled))

  # # CALCULATE Q

  d_vals_cctbx = d_vals_cctbx.as_1d()
  g_vals_cctbx = g_vals_cctbx.as_1d()
  probe_mask_cctbx_double = probe_mask_cctbx.as_1d().as_double()
  q_cctbx = []
  for atomi in range(n_atoms):
    inds = nd_to_1d_indices(
        (None, atomi, None), (n_shells, n_atoms, n_probes)
    )
    # inds = optimized_nd_to_1d_indices(atomi,(n_shells,n_atoms,n_probes))
    inds = flex.uint32(inds)
    d_row = d_vals_cctbx.select(inds)
    g_row = g_vals_cctbx.select(inds)
    mask = probe_mask_cctbx.select(inds)

    d = d_row.select(mask)
    g = g_row.select(mask)
    qval = flex.linear_correlation(d, g).coefficient()
    q_cctbx.append(qval)

  q = flex.double(q_cctbx)
  return q
