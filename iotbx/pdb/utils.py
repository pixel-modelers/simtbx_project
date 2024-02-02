from __future__ import absolute_import, division, print_function
import string
from itertools import product
from six.moves import range
import sys

class generate_n_char_string:
  """ Iterator to generate strings of length n_chars, using upper-case,
    lower-case and numbers as desired.
    Allows specialty sets of characters as well

  parameters:
    n_chars:  length of string to produce
    include_upper:  include upper-case letters
    include_lower:  include lower-case letters
    include_numbers:  include numbers
    include_special_chars: include special characters:
       []_,.;:"&<>()/\{}'`~!@#$%*|+-
    end_with_tilde:  return n_chars - 1 plus the character "~"
    reverse_order:  reverse the order so numbers 9-0, lower case z-a,
                  upper case Z-A

  returns:
    n_char string, new string on every next()
    None if no more strings to return

  Tested in iotbx/regression/tst_generate_n_char_string.py

  """
  def __init__(self, n_chars = 1,
      include_upper = True,
      include_lower = True,
      include_numbers = True,
      include_special_chars = False,
      end_with_tilde = False,
      reverse_order = False):
    self._end_with_tilde = end_with_tilde
    if self._end_with_tilde:
      self._n_chars = n_chars - 1
    else: # usual
      self._n_chars = n_chars

    all_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    all_chars_lc = all_chars.lower()
    all_numbers = '0123456789'
    special_characters = """[]_,.;:"&<>()\/\{}'`~!@#$%*|+-"""
    self._tilde = """~"""

    self._all_everything = ""
    if include_upper:
       self._all_everything += all_chars
    if include_lower:
       self._all_everything += all_chars_lc
    if include_numbers:
       self._all_everything += all_numbers
    if include_special_chars:
       self._all_everything += special_characters

    if reverse_order:
      # Use them in reverse order
      x = []
      for i in self._all_everything:
        x.append(i)
      x.reverse()
      self._all_everything = "".join(x)

    self._n = len(self._all_everything)
    self._indices = []
    for k in range(self._n_chars):
      self._indices.append(0)
  def next(self):
    # Write out current text based on current indices
    value = ""
    for k in range(self._n_chars):
      value += self._all_everything[self._indices[k]]

    # Update indices
    for kk in range(self._n_chars):
      k = self._n_chars - kk - 1 # from last index to first
      self._indices[k] += 1
      if self._indices[k] < self._n: #  current index is in range
        break
      elif k == 0: # current index is out of range and is first index
        return None # no more available
      else: # current index is out of range but is not first index
        self._indices[k] = 0
    if self._end_with_tilde:
      return value + self._tilde
    else: # usual
      return value


def __permutations(iterable, r=None): # XXX This may go to libtbx or scitbx
  pool = tuple(iterable)
  n = len(pool)
  r = n if r is None else r
  for indices in product(range(n), repeat=r):
    if(len(indices) == r):
      yield tuple(pool[i] for i in indices)

def all_chain_ids():
  """
  Test is in
  iotbx/regression/tst_all_chain_ids.py
  There should not be leading whitespace for one letter chain ids.

  Returns all possible 2 character chain IDs for PDB format files.
  In general, returns single character chains first.
  Also tries to use all combinations of uppercase/numbers before going to lower case.
  """
  chars = string.ascii_uppercase+string.digits
  lowerchars = string.ascii_lowercase
  both_char_upper = __permutations(iterable = chars, r = 2)
  second_char_lower = product(chars, lowerchars)
  first_char_lower = product(lowerchars, chars)
  both_char_lower = __permutations(iterable = lowerchars, r = 2)
  # result = [" "+c for c in chars]+[" "+c for c in lowerchars]+\
  result = [c for c in chars]+[c for c in lowerchars]+\
         ["".join(p) for p in both_char_upper]+\
         ["".join(p) for p in second_char_lower]+\
         ["".join(p) for p in first_char_lower]+\
         ["".join(p) for p in both_char_lower]
  return result

def all_label_asym_ids(maximum_length=4):
  chars = string.ascii_uppercase
  rc = ["".join(c) for c in chars]
  for r in range(2, maximum_length+1):
    char_upper = __permutations(iterable = chars, r = r)
    rc += ["".join(p) for p in char_upper]
  return rc

def get_input_model_file_name_from_params(params):
  if not params:
    return ""

  input_scopes = [None, 'input_files','map_model','input']
  input_file_types = ['pdb_in','model','input_model','fixed_model',
     'moving_model','fixed_pdb','moving_pdb','search_model']
  file_name = ""
  for s in input_scopes:
    for x in input_file_types:
      if file_name: break
      if s is None:
        file_name = getattr(params,x,None)
      elif hasattr(params,s):
        ss = getattr(params,s)
        file_name = getattr(ss,x,None)
  if type(file_name) in [type([1,2,3]),type((1,2,3))]:
    if file_name:
      file_name = file_name[0]
    else:
      file_name = ''
  return file_name

def target_output_format_in_params(params):
  """
  Find and return value of target_output_format parameter that may be under
  output_files or output phil scope.
  """
  for x in ['output_files','output']:
    if params and hasattr(params,x) and \
         hasattr(getattr(params,x),'target_output_format'):
      target_output_format = getattr(getattr(params,x),'target_output_format')
      if target_output_format in [None,'','None']:
        target_output_format = None
      return target_output_format
  else:
    return None

def get_target_output_format_from_file_name(file_name,
   default = None):
  if file_name:
    import os
    path, ext = os.path.splitext(file_name)
    if ext == '.pdb':
      target_output_format = 'pdb'
    elif ext == '.cif':
      target_output_format = 'mmcif'
    else:
      target_output_format = default
  else:
    target_output_format = default
  return target_output_format

def move_down_scope_to_input_files(params, levels = 3):
  """
  Find one of the scope names from target_scopes below in params.
  levels - depth of the nested scopes to look in.
  returns first suitable name/scope.
  """
  target_scopes = ['input_files','output','output_files','map_model',]
  if levels < 0:
    return None
  for t in target_scopes:
    if hasattr(params, t):
      return params
  for x in dir(params):
    if x.startswith("_"): continue
    new_p = move_down_scope_to_input_files(getattr(params, x),
      levels = levels - 1)
    if new_p:
      return new_p
  return None

def set_target_output_format_in_params(params,
      file_name = None, default = 'pdb', target_output_format = None,
      out = sys.stdout, quiet = True):
  """
  Find target_output_format parameter in params and:
  - If it is not None, leave it as it was
  - If it is None:
    - set it to extension of file_name or default;
  Return what was set - "pdb" or "mmcif"
  """
  params = move_down_scope_to_input_files(params)
  # Now params pointing out at the correct level of phil scope, the one
  # containing output_files.target_output_format.
  # Note that level to which params points outside this function is not changed.
  # At the same time the value of output_files.target_output_format outside
  # WILL be changed by statements below.

  # Do we have it set already:
  target_output_format = target_output_format_in_params(params)
  if not target_output_format:
    if not file_name:
      file_name = get_input_model_file_name_from_params(params)
    target_output_format = get_target_output_format_from_file_name(
      file_name, default = default)

  # set value in params.output_files.target_output_format
  if hasattr(params,'output_files') and \
       hasattr(params.output_files,'target_output_format'):
     params.output_files.target_output_format = target_output_format
  elif hasattr(params,'output') and \
       hasattr(params.output,'target_output_format'):
     params.output.target_output_format = target_output_format

  # print result
  if not quiet:
    print("Target output format will be: %s" %(target_output_format),
      file = out)

  # Return value
  return target_output_format

def catenate_segment_onto_chain(model_chain, s2, gap = 1,
   keep_numbers = False, insertion_chain = None):
  '''  catenate residues from s2 onto model_chain'''
  from iotbx.pdb import resseq_encode
  if not model_chain:
    return
  if not insertion_chain:
    s2_as_ph = s2.get_hierarchy()
    if not s2_as_ph.overall_counts().n_residues > 0:
      return
    insertion_chain = s2.get_hierarchy().models()[0].chains()[0]
  new_segid = model_chain.residue_groups(
     )[0].atom_groups()[0].atoms()[0].segid
  highest_resseq = None
  if len(model_chain.residue_groups()) > 0:
    highest_resseq = model_chain.residue_groups()[0].resseq_as_int()
  for rg in model_chain.residue_groups():
    rg_resseq = rg.resseq_as_int()
    highest_resseq=max(highest_resseq,rg_resseq)
  resseq_as_int = highest_resseq + (gap - 1)
  for rg in insertion_chain.residue_groups():
    resseq_as_int += 1
    rg_copy = rg.detached_copy()
    if (not keep_numbers):
      rg_copy.resseq = resseq_encode(resseq_as_int)
    model_chain.append_residue_group(
       residue_group = rg_copy)
    rg_copy.link_to_previous = True # Required
    for ag in rg_copy.atom_groups():
      for atom in ag.atoms():
        awl = atom.fetch_labels()
        atom.segid = new_segid

def add_hierarchies(ph_list, create_new_chain_ids_if_necessary = True):
  new_ph_list = []
  for ph in ph_list:
    if ph:
      new_ph_list.append(ph)
  if not new_ph_list:
    return None
  ph = ph_list[0]
  for i in range(1, len(ph_list)):
    ph = add_hierarchy(ph, ph_list[i], create_new_chain_ids_if_necessary =
      create_new_chain_ids_if_necessary)
  return ph


def add_hierarchy(s1_ph, s2_ph, create_new_chain_ids_if_necessary = True):
  ''' Add chains from hierarchy s2_ph to existing hierarchy s1_ph'''
  if not s1_ph:
    return s2_ph
  s1_ph = s1_ph.deep_copy()
  if not s2_ph:
    return s1_ph
  existing_chain_ids = s1_ph.chain_ids()
  for model_mm_2 in s2_ph.models()[:1]:
    for chain in model_mm_2.chains():
      if chain.id in existing_chain_ids: # duplicate chains in add_model
        if not create_new_chain_ids_if_necessary:
          # append to existing chain
          existing_chain = get_chain(s1_ph, chain_id = chain.id)
          catenate_segment_onto_chain(existing_chain, None, gap = 1,
               keep_numbers = True, insertion_chain= chain.detached_copy())
          continue
        else:
          chain.id = get_new_chain_id(existing_chain_ids)
      new_chain = chain.detached_copy()
      existing_chain_ids.append(chain.id)
      for model_mm in s1_ph.models()[:1]:
        model_mm.append_chain(new_chain)
  return s1_ph

def get_chain(s1_ph, chain_id = None):
  for model in s1_ph.models():
    for chain in model.chains():
      if chain.id == chain_id:
        return chain

def lines_are_really_text(lines):
  if lines and type(lines) in (type('abc'), type(b'abc')):
    return True
  else:
    return False

def get_lines(text = None, file_name = None, lines = None):
    import os
    if lines and lines_are_really_text(lines):
      text = lines
    elif lines:
      text = "\n".join(lines)
    elif file_name and os.path.isfile(file_name):
      text = open(file_name).read()
    if not text:
      text = ""
    # Python 3 read fix
    # =======================================================================
    if sys.version_info.major == 3 and type(text) == type(b'abc'):
      text = text.decode("utf-8")
    # =======================================================================
    from cctbx.array_family import flex
    return flex.split_lines(text)

def check_for_missing_elements(hierarchy, file_name = None):
    atoms = hierarchy.atoms()
    elements = atoms.extract_element().strip()
    if (not elements.all_ne("")):
      n_missing = elements.count("")
      missing_list = []
      for atom in list(atoms):
        if not atom.element.strip():
          missing_list.append(atom.format_atom_record())

      raise AssertionError(
        "Uninterpretable elements for %d atoms%s. \n" %(n_missing,
        "" if not file_name else " in '%s'" %(file_name))+
        "Up to 10 listed below: \n%s" % ("\n".join(missing_list[:10])))

def get_pdb_info(text = None, file_name = None, lines = None,
     check_pseudo = False, return_pdb_hierarchy = False,
     return_group_args = False,
     allow_incorrect_spacing = False):
  ''' Get a pdb_input object from pdb or mmcif file, construct a
      hierarchy, check the hierarchy for missing elements and fill them
      in if from pdb.  Return group_args object with
      hierarchy, pdb_input, and crystal_symmetry.

      If text has no atoms, returns empty hierarchy (Note: iotbx.pdb.input
      returns an empty hierarchy in this case if supplied with PDB input
      and raises an exception if supplied with mmCIF input without atoms).

      This method is preferred over get_pdb_input and get_pdb_hierarchy
      as a method for robust reading of pdb/mmcif files because it
      generates the hierarchy only once.  If you run get_pdb_input and then
      construct the hierarchy it is generated twice.
  '''
  return get_pdb_input(text = text, file_name = file_name,
     lines = lines, check_pseudo = check_pseudo,
     allow_incorrect_spacing = allow_incorrect_spacing,
     return_group_args = True)

def get_pdb_hierarchy(text=None, file_name = None,
     lines = None, check_pseudo = None,
     allow_incorrect_spacing = False):
  ''' Get pdb_input object and construct hierarchy.  Normally use instead
      info =  get_pdb_info and then take info.hierarchy so that you have
      the pdb_input object and crystal_symmetry available as well
  '''
  return get_pdb_input(text = text, file_name = file_name, lines = lines,
     check_pseudo = check_pseudo, return_pdb_hierarchy = True,
     allow_incorrect_spacing = allow_incorrect_spacing)


def get_pdb_input(text = None, file_name = None, lines = None,
    check_pseudo = False, return_pdb_hierarchy = False,
    return_group_args = False,
     allow_incorrect_spacing = False):

  ''' Get a pdb_input object from pdb or mmcif file, construct a
      hierarchy, check the hierarchy for missing elements and fill them
      in if from pdb.  Return hierarchy, pdb_input, or
      group_args object with hierarchy, pdb_input, and crystal_symmetry.

      Normally use instead the info = get_pdb_info method and then
      you have hierarchy, pdb_input and crystal_symmetry all available

  '''
  lines = get_lines(text = text, file_name = file_name, lines = lines)

  # Get input object as pdb_inp
  import iotbx.pdb
  pdb_inp = iotbx.pdb.input(source_info=None,lines=lines)

  # Guess elements if PDB is source input and elements are missing.
  # This is can only be done with PDB files
  # because mmcif files lose the positional information
  # in atom names, so CA cannot be distinguished from Ca (calcium) without
  #  element information in an mmCIF file.

  if type_of_pdb_input(pdb_inp) == 'pdb': # Guess elements if missing for PDB
    ph = try_to_get_hierarchy(pdb_inp)
    ph.guess_chemical_elements(check_pseudo = check_pseudo,
      allow_incorrect_spacing = allow_incorrect_spacing)
    check_for_missing_elements(ph)
  else:
    # Make sure we have an element for each atom
    # try to construct and save empty ph if fails
    ph = try_to_get_hierarchy(pdb_inp)
    check_for_missing_elements(ph)

  # Return what is requested
  if return_group_args:
    from libtbx import group_args
    return group_args(group_args_type = 'hierarchy and pdb_input',
      hierarchy = ph,
      pdb_inp = pdb_inp,
      crystal_symmetry = pdb_inp.crystal_symmetry())

  elif return_pdb_hierarchy:
    return ph
  else:
    return pdb_inp


def set_element_ignoring_spacings(hierarchy):
  ''' Set missing elements ignoring spacings. This allows
   reading a PDB file where there are no elements given and the
   atom names are not justified properly. Intended for hetero atoms 
   even if they are not marked as such.  Normally try to set
   elements in normal way first.
  '''

  atoms = hierarchy.atoms()
  elements = atoms.extract_element().strip()
  sel = (elements == "")
  atoms_sel = atoms.select(sel)
  for at in atoms_sel:
    if at.name.startswith(" "):
      at.name = at.name[1:]
    else:
      at.name = " "+at.name[:3]
  atoms.set_chemical_element_simple_if_necessary()

def check_for_pseudo_atoms(hierarchy):
    # Check for special case where PDB input contains pseudo-atoms ZC ZU etc
    atoms = hierarchy.atoms()
    # Do we already have all the elements
    elements = atoms.extract_element().strip()
    if elements.all_ne(""): # all done
      return

    # Are there any pseudo-atoms
    atom_names = atoms.extract_name().strip()
    all_text = "".join(list(atom_names))
    if all_text.find("Z") < 0: # no pseudo-atoms
      return

    # contains some pseudo-atoms ZC ZU etc. Get their elements if necessary
    for atom in atoms:
      if atom.element.replace(" ","") == '':
        # Missing element not fixed by set_chemical_element_simple_if_necessary
        #  take first non-Z, non-blank character
        for c in atom.name.replace("Z","").replace(" ",""):
          if c.isalpha():
            atom.element=c
            break

def type_of_pdb_input(pdb_inp):
  format_type = None
  if not pdb_inp:
    return format_type
  else:
    s = str(type(pdb_inp))
    if s.find("cif") > 0:
      format_type = "mmcif"
    elif s.find("pdb") > 0:
      format_type = "pdb"
    return format_type

def try_to_get_hierarchy(pdb_inp):
    try:
      return pdb_inp.construct_hierarchy()
    except Exception as e: # nothing there
      if str(e).find("something is not present") > -1:  # was empty hierarchy
        # NOTE this text is in modules/cctbx_project/iotbx/pdb/mmcif.py
        # If it changes, change it here too.
        from iotbx.pdb import hierarchy
        ph = hierarchy.root()
        return ph
      else:  # Stop and ask developers to check code
        ph_text = "\n".join(lines)
        text = """The above hierarchy could not be read. If it is just empty,
         please ask developers to check that the
         text "something is not present" is used in
         modules/cctbx_project/iotbx/pdb/mmcif.py as part of the assertion that
         atoms are present.  Modify this text to match the assertion if
         necessary"""
        raise Sorry(ph_text+"\n"+text+"\n"+str(e))

def add_model(s1, s2, create_new_chain_ids_if_necessary = True):
  ''' Add chains from s2 to existing s1 to create new composite model'''
  if not s1:
    s2.reset_after_changing_hierarchy()
    return s2
  s1.add_crystal_symmetry_if_necessary()
  s1 = s1.deep_copy()
  if not s2:
    s1.reset_after_changing_hierarchy()
    return s1
  s1_ph = s1.get_hierarchy() # working hierarchy
  existing_chain_ids = []
  from mmtbx.secondary_structure.find_ss_from_ca import get_new_chain_id
  for model_mm1 in s1_ph.models()[:1]:
    for chain in model_mm1.chains():
      if not chain.id.strip():
        chain.id = get_new_chain_id(existing_chain_ids)
  existing_chain_ids = s1_ph.chain_ids()
  for model_mm_2 in s2.get_hierarchy().models()[:1]:
    for chain in model_mm_2.chains():
      if not chain.id.strip():
        chain.id = get_new_chain_id(existing_chain_ids)

      if chain.id in existing_chain_ids: # duplicate chains in add_model
        if not create_new_chain_ids_if_necessary:
          # append to existing chain
          from iotbx.pdb.utils import get_chain
          from iotbx.pdb.utils import catenate_segment_onto_chain
          existing_chain = get_chain(s1_ph, chain_id = chain.id)
          catenate_segment_onto_chain(existing_chain, None, gap = 1,
               keep_numbers = True, insertion_chain= chain.detached_copy())
          continue
        else:
          chain.id = get_new_chain_id(existing_chain_ids)
      new_chain = chain.detached_copy()
      existing_chain_ids.append(chain.id)
      for model_mm in s1_ph.models()[:1]:
        model_mm.append_chain(new_chain)
  s1.reset_after_changing_hierarchy()

  # Handle model.info().numbering_dict if present
  if hasattr(s1,'info') and s1.info().get('numbering_dict') and \
     hasattr(s2,'info') and s2.info().get('numbering_dict'):
    s1.info().numbering_dict.add_from_other(s2.info().numbering_dict)
  return s1

def catenate_segments(s1, s2, gap = 1,
   keep_numbers = False):
  '''
    catenate two models and renumber starting with first residue of s1
    if gap is set, start s2  gap residue numbers higher than the end of s1
    if keep_numbers is set, just keep all the residue numbers
  '''
  s1 = s1.deep_copy()
  s1 = s1.apply_selection_string("not (name OXT)") # get rid of these
  s1_ph = s1.get_hierarchy() # working hierarchy
  for model_mm in s1_ph.models()[:1]:
    for model_chain in model_mm.chains()[:1]:
        from iotbx.pdb.utils import catenate_segment_onto_chain
        catenate_segment_onto_chain(
          model_chain,
          s2,
          gap = gap,
          keep_numbers = keep_numbers)
  s1.reset_after_changing_hierarchy()
  return s1
def catenate_segment_onto_chain(model_chain, s2, gap = 1,
   keep_numbers = False, insertion_chain = None):
  '''  catenate residues from s2 onto model_chain'''
  from iotbx.pdb import resseq_encode
  if not model_chain:
    return
  if not insertion_chain:
    s2_as_ph = s2.get_hierarchy()
    if not s2_as_ph.overall_counts().n_residues > 0:
      return
    insertion_chain = s2.get_hierarchy().models()[0].chains()[0]
  new_segid = model_chain.residue_groups(
     )[0].atom_groups()[0].atoms()[0].segid
  highest_resseq = None
  if len(model_chain.residue_groups()) > 0:
    highest_resseq = model_chain.residue_groups()[0].resseq_as_int()
  for rg in model_chain.residue_groups():
    rg_resseq = rg.resseq_as_int()
    highest_resseq=max(highest_resseq,rg_resseq)
  resseq_as_int = highest_resseq + (gap - 1)
  for rg in insertion_chain.residue_groups():
    resseq_as_int += 1
    rg_copy = rg.detached_copy()
    if (not keep_numbers):
      rg_copy.resseq = resseq_encode(resseq_as_int)
    model_chain.append_residue_group(
       residue_group = rg_copy)
    rg_copy.link_to_previous = True # Required
    for ag in rg_copy.atom_groups():
      for atom in ag.atoms():
        awl = atom.fetch_labels()
        atom.segid = new_segid

def simple_combine(model_list,
    create_new_chain_ids_if_necessary = True):
  ''' Method to combine the chains in a set of models to create a new
  model with all the chains.
  param: model_list:  list of model objects
  param: create_new_chain_ids_if_necessary:  If True (default), if a
          model has a duplicate chain ID, create a new one and rename it
  returns:  first model in model_list with all chains from all models.
          NOTE: first model in model_list is modified by this method. Make
          a deep_copy before hand if you want to keep it.
  '''

  model = None
  for m in model_list:
    if not model:
      model = m  # ZZZ why cannot this be a deep_copy?
    else:
      model = add_model(model, m,
         create_new_chain_ids_if_necessary = create_new_chain_ids_if_necessary)
  return model

def get_cif_or_pdb_file_if_present(file_name):
   ''' Identify whether a file with the name file_name or with
   alternate extensions replacing pdb/cif is present.
   If file_name is present, return file_name.
   If not, and alternative is present, return alternative file name
   Otherwise return empty string
   '''

   import os
   if file_name and os.path.isfile(file_name): # if it is present, take it
     return file_name
   # Otherwise, look for pdb or cif versions of this file
   p,e = os.path.splitext(file_name)
   e_pdb = e.replace("cif","pdb")
   e_cif = e.replace("pdb","cif")
   pdb_file = "%s%s" %(p,e_pdb)
   cif_file = "%s%s" %(p,e_cif)
   if os.path.isfile(pdb_file):
     return pdb_file
   elif os.path.isfile(cif_file):
     return cif_file
   else:
     return "" # return empty string so os.path.isfile(return_value) works

class numbering_dict:
  ''' Set up a dict that keeps track of chain ID, residue ID and icode for
    residues relative to their initial values
    dict with keys of original residue chain ID, resseq, icode
    inverse_dict with keys of current, values of original
  '''
  def __init__(self, m):
    self.file_name = m.info().file_name
    self.ph = m.get_hierarchy().deep_copy()
    self.dd = self.get_dict(m) # keys are original, values current
    self.get_inverse_dict()  # keys are current, values original

  def show_summary(self):
    print("Summary of numbering dict for %s: " %(self.file_name))
    for model in self.ph.models():
      for chain in model.chains():
        for rg in chain.residue_groups():
          for conformer in rg.conformers():
            r = conformer.only_residue()
            key = self.get_key(r)
            print(key, self.original_key_from_current(key))

  def update(self, new_m):
    ''' Update the dicts to refer to new current model
    new_m must be similar hierarchy to previous current model
    '''
    new_dd = {}
    new_ph = new_m.get_hierarchy()
    ph = self.ph


    for new_model, model in zip(new_ph.models()[:1], ph.models()[:1]):
      for new_chain, chain in zip(new_model.chains(), model.chains()):
        for new_rg, rg in zip(new_chain.residue_groups(), chain.residue_groups()):
          for new_conformer, conformer in zip(
             new_rg.conformers(),
             rg.conformers()):
            new_r = new_conformer.only_residue()
            r = conformer.only_residue()

          new_key = self.get_key(new_r)
          key = self.get_key(r)
          original_key = self.original_key_from_current(key)
          new_dd[original_key] = new_key
    self.dd = new_dd
    self.ph = new_ph.deep_copy()
    self.get_inverse_dict()

  def get_original_key_list_from_atoms(self, atoms):
   original_key_list = []
   for key in self.get_key_list_from_atoms(atoms):
     original_key_list.append(self.original_key_from_current(key))
   return original_key_list

  def get_key_list_from_atoms(self, atoms):
    key_list = []
    for a in atoms:
      rg = a.parent().parent()
      for c in rg.conformers():
        r = c.only_residue()
        key_list.append(self.get_key(r))
    return key_list

  def add_from_other(self, other):
    for key in other.dd.keys():
      self.dd[key] = other.dd[key]
    self.get_inverse_dict()

  def get_key(self, r):
    conformer = r.parent()
    chain = conformer.parent()
    return "%s %s %s %s" %(r.resname, chain.id, r.resseq, r.icode)

  def current_key_from_current_r(self,r):
    return self.get_key(r)

  def original_key_from_current_r(self,r):
    key = self.get_key(r)
    return self.original_key_from_current(key)

  def original_key_from_current(self, key):
    return self.inverse_dict[key]

  def current_key_from_original(self, key):
    return self.dd[key]

  def get_inverse_dict(self):
    self.inverse_dict = {}
    for key in self.dd.keys():
      self.inverse_dict[self.dd[key]] = key

  def get_dict(self, m):
    dd = {}
    ph = m.get_hierarchy()
    for model in ph.models()[:1]:
      for chain in model.chains():
        for rg in chain.residue_groups():
          for conformer in rg.conformers():
            r = conformer.only_residue()
            key = self.get_key(r)
            dd[key] = key
    return dd

if __name__ == '__main__':
  import time
  l=0
  p=1
  for r in range(1,5):
    t0=time.time()
    rc = all_label_asym_ids(maximum_length=r)
    p*=26
    l+=p
    print('%7d %7d %7d %5s %0.3fs' % (l,p,len(rc), rc[-1], time.time()-t0))
    assert len(rc)==l
