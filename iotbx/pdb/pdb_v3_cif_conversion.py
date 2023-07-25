from __future__ import absolute_import, division, print_function

'''
pdb_v3_cif_conversion.py

Methods to convert between a hierarchy object and a pdb_v3 compatible string.

Rationale: Hierarchy object and mmcif representations can contain
  chain ID values with n-characters and residue names with 3 or 5
  characters.  PDB format only allows 2 chars for chain ID and 3 for
  residue names.

Approach: Convert all non-pdb_v3-compliant chain ID and residue names
  to suitable number of characters and save the conversion information
  as a conversion_info object and as REMARK records in PDB string
  representations of the hierarchy.

Examples of typical uses:

A. Write a pdb_v3 compatible string with conversion information in REMARK
   records from any hierarchy (ph):
   NOTE: any kw and args for as_pdb_string() can be supplied

  from iotbx.pdb.pdb_v3_cif_conversion import hierarchy_as_pdb_v3_string
  pdb_v3_string =  hierarchy_as_pdb_v3_string(ph)

B. Read a pdb_v3 compatible string (pdb_v3_string) with conversion
   information in REMARK records and convert to a hierarchy (inverse of A).
   NOTE: same function will read any mmcif string as well.

  from iotbx.pdb.pdb_v3_cif_conversion import pdb_or_mmcif_string_as_hierarchy
  ph = pdb_or_mmcif_string_as_hierarchy(pdb_v3_string)

C. Get conversion info from any hierarchy (ph):

  from iotbx.pdb.pdb_v3_cif_conversion import pdb_v3_cif_conversion
  conversion_info = pdb_v3_cif_conversion(ph)

D. Get conversion info as REMARK string

  from iotbx.pdb.pdb_v3_cif_conversion import pdb_v3_cif_conversion
  remarks_string = pdb_v3_cif_conversion(ph).conversion_as_remark_string()

E. Convert a pdb_v3 compatible hierarchy to a full hierarchy with
   conversion information in conversion_info. This approach can be
   used to (1) save conversion information from a hierarchy,
   (2) write a pdb_v3 file, (3) do something with the pdb_v3 file that loses
   the header information, (4) read back the pdb_v3 file that does not have
   REMARK records, and (5) restore the original structure in the new
   hierarchy.

  from iotbx.pdb.pdb_v3_cif_conversion import pdb_v3_cif_conversion
  from iotbx.pdb.pdb_v3_cif_conversion import hierarchy_as_pdb_v3_string
  from iotbx.pdb.pdb_v3_cif_conversion import pdb_or_mmcif_string_as_hierarchy

  # Get conversion information
  conversion_info = pdb_v3_cif_conversion(ph)

  # Get a pdb_v3 string with no remarks
  pdb_v3_string =  hierarchy_as_pdb_v3_string(ph)
  pdb_v3_string_no_remarks = remove_remarks(pdb_v3_string)

  # convert back to hierarchy (this can be a new pdb string obtained
  #  after manipulations of the model but with residue names and chain id
  #  values matching the pdb_v3_string)
  ph = pdb_or_mmcif_string_as_hierarchy(pdb_v3_string_no_remarks)

  # Apply the conversions to obtain a full representation in ph
  conversion_info.convert_hierarchy_to_full_representation(ph)
  '''

def hierarchy_as_pdb_v3_string(ph, *args, **kw):
  '''Convert a hierarchy into a pdb_v3 compatible string, with any
     conversion information written as REMARK records

    parameters:
      ph: hierarchy object
      args, kw: any args and kw suitable for the hierarchy
          method ph.as_pdb_string()

    returns:  string
  '''

  conversion_info = pdb_v3_cif_conversion(hierarchy = ph)
  if (not conversion_info.conversion_required()):
    return ph.as_pdb_string(*args, **kw)
  else:
    ph_pdb_v3 = ph.deep_copy()
    conversion_info.convert_hierarchy_to_pdb_v3_representation(ph_pdb_v3)
    remarks_string = conversion_info.conversion_as_remark_string()
    pdb_v3_string = ph_pdb_v3.as_pdb_string(*args, **kw)
    full_string = remarks_string + pdb_v3_string
    return full_string

def pdb_or_mmcif_string_as_hierarchy(pdb_or_mmcif_string,
       conversion_info = None):
  '''Convert an mmcif string or a pdb_v3 compatible string into a
      hierarchy object, using any conversion information written as
      REMARK records in the pdb_v3 string, or using any supplied
      conversion information.

    parameters:
      pdb_or_mmcif_string: mmcif string or a pdb_v3 compatible string
      conversion_info: optional pdb_v3_cif_conversion object to apply

    returns: hierarchy
  '''
  import iotbx.pdb
  from iotbx.pdb.pdb_v3_cif_conversion import pdb_v3_cif_conversion
  inp = iotbx.pdb.input(lines=pdb_or_mmcif_string, source_info=None)
  remark_string = "\n".join(inp.remark_section())
  if (not conversion_info):
    conversion_info = pdb_v3_cif_conversion()
    conversion_info.set_conversion_tables_from_remarks_records(
      remarks_records = remark_string.splitlines())
  assert conversion_info.is_initialized()

  # Get the hierarchy
  ph = inp.construct_hierarchy()

  # Determine if this is already in full format
  if pdb_v3_cif_conversion(ph).conversion_required(): # already set
    assert not conversion_info.conversion_required(), \
      "Cannot apply pdb_v3 conversions to a hierarchy that is not pdb_v3"
    return ph
  elif conversion_info.conversion_required(): # convert it
    conversion_info.convert_hierarchy_to_full_representation(ph)
    return ph
  else: # nothing to convert
    return ph

class pdb_v3_cif_conversion:
  ''' Class to generate and save pdb_v3 representation of 5-character
    residue names and n-character chain IDs. Used to convert between
    pdb_v3 and mmcif formatting.

    NOTE 1: marked as self._is_initialized when values are available
    NOTE 2: hierarchy object that has been converted to pdb_v3 compatible
    will be marked with the attribute
      self._is_pdb_v3_representation=True


    To modify these tables to add another field to check:
    1. Add new field to self._keys and self._max_chars_dict
    2. Add new methods like "def _unique_chain_ids_from_hierarchy"
    3. Use these new methods in "def _set_up_conversion_table"
    4. Add code at "Modify hierarchy here to convert to pdb_v3"
    5. Add code at "Modify hierarchy here to convert from pdb_v3"
    6. Add code to regression test at iotbx/regression/tst_hierarchy_pdb_v3.py
    '''

  def __init__(self, hierarchy = None):
    ''' Identify all unique chain_ids and residue names that are not compatible
        with pdb_v3. Generate dictionary relating original names and
        compatible names and for going backwards.

    parameters:  iotbx.pdb.hierarchy object (required)

    returns:  None

    '''


    # Fields in hierarchy that are limited in number of characters in pdb_v3
    self._keys = ['chain_id', 'resname']
    self._max_chars_dict = {'chain_id':2, 'resname':3}

    self._is_initialized = False

    if hierarchy is None:
      self._conversion_table_info_dict = None
      self._conversion_required = None
      return

    # Set up conversion tables
    self._conversion_table_info_dict = {}

    # Flag that indicates if any conversion is necessary
    self._conversion_required = False

    for key in self._keys:
      self._set_up_conversion_table(key, hierarchy)

    self._is_initialized = True

  def is_initialized(self):

    '''Public method to return True if this is initialized
    parameters:  None
    returns: True if initialized
    '''
    return self._is_initialized

  def conversion_required(self):

    '''Public method to return True if conversion for pdb_v3 is necessary
    parameters:  None
    returns: True if conversion is necessary
    '''
    assert self.is_initialized(), "Need to initialize"
    return self._conversion_required

  def conversion_as_remark_string(self):
    '''Public method to return a PDB REMARK string representing all the
    conversions that are necessary
    '''

    assert self.is_initialized(), "Need to initialize"

    if not self.conversion_required():
      return ""  # No info needed


    from six.moves import cStringIO as StringIO
    f = StringIO()
    print(
       "REMARK   PDB_V3_CONVERSION  CONVERSIONS MADE FOR PDB_V3 COMPATIBILITY",
           file = f)
    for key in self._keys:
      info =  self._conversion_table_info_dict[key]
      if info:
        for full_text, pdb_v3_text in zip(
           info.full_representation_list,
           info.pdb_v3_representation_list,
           ):
          print(
            "REMARK   PDB_V3_CONVERSION  %s: %s  PDB_V3_TEXT: %s" %(
              key.upper(),
              full_text,
              pdb_v3_text),
            file = f)
    print(file = f)
    return f.getvalue()


  def convert_hierarchy_to_pdb_v3_representation(self, hierarchy):

    '''Public method to convert a hierarchy in place to pdb_v3 compatible
       hierarchy using information in self._conversion_table_info_dict
    parameters: hierarchy (modified in place)
    output: None

    '''

    assert self.is_initialized(), "Need to initialize"
    assert hierarchy is not None, "Need hierarchy for conversion"

    if hasattr(hierarchy, '_is_pdb_v3_representation') and (
        hierarchy._is_pdb_v3_representation):
      return # nothing to do because it was already converted

    if not self.conversion_required():
      return # nothing to do because no conversion is necessary

    # Modify hierarchy here to convert to pdb_v3

    for model in hierarchy.models():
      for chain in model.chains():
        new_id = self._get_pdb_v3_text_from_full_text(
           key = 'chain_id',
           full_text = chain.id)
        if new_id and new_id != chain.id:
          chain.id = new_id  # Modify chain ID here

        for residue_group in chain.residue_groups():
          for atom_group in residue_group.atom_groups():
            new_resname = self._get_pdb_v3_text_from_full_text('resname',
                atom_group.resname)
            if new_resname and (new_resname != atom_group.resname):
              atom_group.resname = new_resname  # Modify residue name here

    hierarchy._is_pdb_v3_representation = True

  def convert_hierarchy_to_full_representation(self, hierarchy):
    '''Public method to convert a hierarchy in place from pdb_v3 compatible
       hierarchy using information in self._conversion_table_info_dict
    parameters: hierarchy (modified in place)
    output: None

    '''
    assert hierarchy is not None, "Need hierarchy for conversion"
    assert self.is_initialized(), "Need to initialize"

    if hasattr(hierarchy, '_is_pdb_v3_representation') and (
        not self.hierarchy_is_full_representation):
      return # nothing to do because it was already converted

    if not self.conversion_required():
      return # nothing to do because no conversion is necessary

    # Modify hierarchy here to convert from pdb_v3

    for model in hierarchy.models():
      for chain in model.chains():
        new_id = self._get_full_text_from_pdb_v3_text(
          key = 'chain_id',
          pdb_v3_text = chain.id)
        if new_id and new_id != chain.id:
          chain.id = new_id  # Modify chain_id here

        for residue_group in chain.residue_groups():
          for atom_group in residue_group.atom_groups():
            new_resname = self._get_full_text_from_pdb_v3_text(
              key = 'resname',
              pdb_v3_text = atom_group.resname)
            if new_resname and (new_resname != atom_group.resname):
              atom_group.resname = new_resname # Modify residue name here

    hierarchy._is_pdb_v3_representation = False

  def set_conversion_tables_from_remarks_records(self, remarks_records):
    ''' Public method to set conversion tables based on remarks records
        written in standard form as by this class

    parameters:
      remarks_records:  list of lines, containing REMARK lines with information
                        conversion_as_remark_string
    returns: None
    '''

    self._is_initialized = True

    if not remarks_records:
      return # nothing to do

    self._conversion_required = False

    full_representation_list_dict = {}
    pdb_v3_representation_list_dict = {}
    for key in self._keys:
      full_representation_list_dict[key] = []
      pdb_v3_representation_list_dict[key] = []

    for line in remarks_records:
      if not line: continue
      if not line.startswith("REMARK"): continue
      spl = line.split()
      if len(spl) != 6: continue
      if not spl[1] == "PDB_V3_CONVERSION": continue
      key = spl[2].lower()[:-1] # take off ":"
      if not key in self._keys: continue
      full = spl[3]
      pdb_v3 = spl[5]
      full_representation_list_dict[key].append(full)
      pdb_v3_representation_list_dict[key].append(pdb_v3)

      # there was something needing conversion
      self._conversion_required = True

    self._is_initialized = True

    if not self._conversion_required: # nothing to do
      return

    self._conversion_table_info_dict = {}
    from libtbx import group_args
    for key in self._keys:
      self._conversion_table_info_dict[key] = group_args(
        group_args_type = 'conversion tables for %s' %(key),
        full_representation_list = full_representation_list_dict[key],
        pdb_v3_representation_list =  pdb_v3_representation_list_dict[key])


  def _set_up_conversion_table(self, key, hierarchy):
    ''' Private method to set up conversion table from a hierarchy for
        field named by key and put it in self._conversion_table_info_dict[key].
        Also set self._conversion_required if conversion is needed.
        also set self._is_initialized'''

    if key == 'chain_id':
      unique_values = self._unique_chain_ids_from_hierarchy(hierarchy)
    elif key == 'resname':
      unique_values = self._unique_resnames_from_hierarchy(hierarchy)
    else:
      raise "NotImplemented"

    max_chars = self._max_chars_dict[key]
    allowed_ids, ids_needing_conversion = self._choose_allowed_ids(
        unique_values,
        max_chars = max_chars)

    pdb_v3_representation_list = self._get_any_pdb_v3_representation(
        ids_needing_conversion, max_chars, exclude_list = allowed_ids)

    if ids_needing_conversion:
      assert len(ids_needing_conversion) == len(pdb_v3_representation_list)

    from libtbx import group_args
    self._conversion_table_info_dict[key] = group_args(
      group_args_type = 'conversion tables for %s' %(key),
      full_representation_list = ids_needing_conversion,
      pdb_v3_representation_list = pdb_v3_representation_list)

    if pdb_v3_representation_list:  # there was something needing conversion
      self._conversion_required = True

  def _unique_chain_ids_from_hierarchy(self, hierarchy):
    ''' Private method to identify all unique chain IDs in a hierarchy
    parameters:  hierarchy
    returns:  list of unique chain ids

    '''
    chain_ids = []
    for model in hierarchy.models():
      for chain in model.chains():
        if (not chain.id in chain_ids):
          chain_ids.append(chain.id)
    return chain_ids

  def _unique_resnames_from_hierarchy(self, hierarchy):
    ''' Private method to identify all unique residue names in a hierarchy
    parameters:  hierarchy
    returns:  list of unique residue names

    '''
    resnames = []
    for model in hierarchy.models():
      for chain in model.chains():
        for residue_group in chain.residue_groups():
          for atom_group in residue_group.atom_groups():
            if (not atom_group.resname in resnames):
              resnames.append(atom_group.resname)
    return resnames

  def _get_any_pdb_v3_representation(self, ids_needing_conversion,
     max_chars, exclude_list = None,):
    '''Private method to try a few ways to generate pdb_v3 representations
      for a set of strings
    parameters:
      ids_needing_conversion:  list of strings to convert
      max_chars:  maximum characters in converted strings
      exclude_list: list of strings not to use as output
    returns:
      pdb_v3_representation_list: list of converted strings, same order and
        length as ids_needing_conversion
    '''

    if (not ids_needing_conversion):
       return [] # ok with nothing in it

    # Try just taking first n_chars of strings...ok if they are all unique
    pdb_v3_representation_list = self._get_pdb_v3_representation(
        ids_needing_conversion, max_chars, exclude_list = exclude_list,
        take_first_n_chars = True)
    if pdb_v3_representation_list:
      return pdb_v3_representation_list

    # Generate unique strings for all the ids needing conversion, preventing
    #   duplications of existing ids
    pdb_v3_representation_list = self._get_pdb_v3_representation(
        ids_needing_conversion, max_chars, exclude_list = exclude_list)
    if pdb_v3_representation_list:
      return pdb_v3_representation_list

    # Failed to get pdb_v3 representation...
    from libtbx.utils import Sorry
    raise Sorry("Unable to generate pdb_v3 representation of %s" %(key))

  def _get_pdb_v3_representation(self, ids, max_chars,
      exclude_list = None, take_first_n_chars = False,):

    '''Private method to try and get pdb_v3 representation of ids that fit in
       max_chars and do not duplicate anything in exclude_list
    parameters:
      ids:  strings to convert
      max_chars:  maximum characters in output
      exclude_list: strings to not include in output
      take_first_n_chars: just take the first max_chars if set

    returns:
      list of converted strings of same order and length as ids, if successful
      otherwise, None
    '''

    pdb_v3_representation_list = []
    for id in ids:
      if take_first_n_chars:  # Just take the first n_chars
        new_id = id[:max_chars]
        if new_id in exclude_list + pdb_v3_representation_list:
          return None # cannot do it this way
      else:  # generate a new id
        new_id = self._get_new_unique_id(id, max_chars,
           exclude_list + pdb_v3_representation_list)
        if not new_id:
          return None # could not do this
      pdb_v3_representation_list.append(new_id)
    return pdb_v3_representation_list

  def _get_new_unique_id(self, id, max_chars, exclude_list):
    ''' Private method to get a unique ID with up to max_chars that is not
    in exclude_list. Start with max_chars and work down and use reverse order
    so as to generally create codes that are unlikely for others to have used.
    '''
    for n_chars_inv in range(max_chars):
      n_chars = max_chars - n_chars_inv
      id = self._get_new_id(n_chars, exclude_list)
      if id:
        return id

  def _get_new_id(self, n_chars, exclude_list):
    ''' Private method to get a unique ID with exactly n_chars that is not
    in exclude_list
    '''
    from iotbx.pdb.utils import generate_n_char_string
    x = generate_n_char_string(n_chars = n_chars,
       reverse_order = True)
    while 1:
      new_id = x.next()
      if (not new_id):
        return None # failed
      elif (not new_id in exclude_list):
        return new_id

  def _choose_allowed_ids(self, unique_values, max_chars):
    ''' Private method to separate unique_values into those that are and
        are not compatible with pdb_v3 (i.e., have max_chars or fewer)
    '''
    allowed = []
    not_allowed = []
    for u in unique_values:
      if self._is_allowed(u, max_chars):
        allowed.append(u)
      else:
        not_allowed.append(u)
    return allowed, not_allowed

  def _is_allowed(self, u, max_chars):
    ''' Private method to identify whether the string u is or is not
        compatible with pdb_v3 (i.e., has max_chars or fewer)
    '''
    if len(u) <= max_chars:
      return True
    else:
      return False


  def _get_conversion_table_info(self, key):
    ''' Private method to return conversion table info for
        specified key (e.g., chain_id, resname)
    '''

    if not key in self._keys:
      return None
    else:
      return self._conversion_table_info_dict[key]

  def _get_full_text_from_pdb_v3_text(self, key = None, pdb_v3_text = None):
    '''Private method to return full text from pdb_v3_text based on
       conversion table

    parameters:
      key: field to convert (e.g., chain_id, resname)
      pdb_v3_text: text to convert from pdb_v3 to full text
    '''

    assert key is not None
    assert pdb_v3_text is not None

    conversion_table_info = self._get_conversion_table_info(key)

    if conversion_table_info and (
        pdb_v3_text in conversion_table_info.pdb_v3_representation_list):
      index = conversion_table_info.pdb_v3_representation_list.index(
        pdb_v3_text)
      full_text = conversion_table_info.full_representation_list[index]
    else:
      full_text = pdb_v3_text

    return full_text

  def _get_pdb_v3_text_from_full_text(self, key = None, full_text = None):
    '''Private method to return pdb_v3 text from full text based on
       conversion table

    parameters:
      key: field to convert (e.g., chain_id, resname)
      full_text: text to convert to pdb_v3
    '''

    assert key is not None
    assert full_text is not None

    conversion_table_info = self._get_conversion_table_info(key)
    if conversion_table_info and (
        full_text in conversion_table_info.full_representation_list):
      index = conversion_table_info.full_representation_list.index(
        full_text)
      pdb_v3_text = conversion_table_info.pdb_v3_representation_list[index]
    else:
      pdb_v3_text = full_text

    # Make sure that the resulting text is allowed in pdb_v3
    assert self._is_allowed(pdb_v3_text, self._max_chars_dict[key])

    return pdb_v3_text
