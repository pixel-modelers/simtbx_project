from __future__ import absolute_import, division, print_function
# LIBTBX_SET_DISPATCHER_NAME phenix.pdb_cif_conversion

info_string = '''

===========================================================================
===========================================================================
CCTBX methods, tools, and strategies for conversion of PDB-format
based methods to mmCIF/PDB compatible methods
2024-01-30 TT
===========================================================================
===========================================================================
       SECTIONS:

   SUMMARY OF RECOMMENDED PROCEDURES
   GOALS
   RECOMMENDED OVERALL APPROACHES
   DETAILED SUGGESTIONS FOR MAKING CODE CIF-COMPLIANT
     USING THE PROGRAM TEMPLATE TO HANDLE PDB/MMCIF INPUT/OUTPUT
     TOOLS AVAILABLE IF YOU CANNOT USE THE PROGRAM TEMPLATE
     REWRITING CODE USING PDB-FORMATTED TEXT
     TOOLS AVAILABLE TO FIND CODE THAT NEEDS TO BE MADE CIF-COMPATIBLE
     CREATING CIF TESTS TO CHECK CODE WITH MODELS THAT CANNOT FIT IN PDB FORMAT
     USING FORWARD-COMPATIBLE PDB FORMAT FOR CODE REQUIRING PDB-FORMATTED TEXT
   DETAILS OF METHODS ADDED FOR WORKING WITH PDB/CIF

===========================================================================
===========================================================================

SUMMARY OF RECOMMENDED PROCEDURES

I.   USE THE PROGRAM TEMPLATE
II.  TOOLS ARE AVAILABLE IF YOU CANNOT USE THE PROGRAM TEMPLATE
III. IF CODE USES PDB-FORMATTED TEXT IT SHOULD BE REWRITTEN
IV.  TOOLS ARE AVAILABLE TO FIND CODE THAT NEEDS TO BE MADE CIF-COMPATIBLE
V.   CREATE CIF TESTS TO CHECK CODE WITH MODELS THAT CANNOT FIT IN PDB FORMAT
VI.  USE FORWARD-COMPATIBLE PDB FORMAT FOR CODE REQUIRING PDB-FORMATTED TEXT

===========================================================================
===========================================================================

GOALS:

I. Read, write and work with models in mmCIF or PDB format interchangeably if
   models fit in PDB format, otherwise read and write only in mmCIF. In
   exceptional circumstances, allow read/write of forward-compatible PDB files
   representing models that do not fit in PDB format.

II. Use the cctbx hierarchy and model objects to contain and work with
   all models

III. Write mmCIF or PDB-formatted files with the rules:
   a. Write as PDB with extension '.pdb' if output model fits in PDB format and,
     1. User specifies PDB format, or,
     2. User does not specify format and either there is no input model or it is
        PDB format
   b. Write as mmCIF, with extension '.cif', if
     1. Output model does not fit in PDB format, or
     2. User specifies mmCIF, or
     3. Input model is mmCIF and User does not specify which to use
===========================================================================
===========================================================================

    RECOMMENDED OVERALL APPROACHES
  (DETAILS IN THE SECTION "DETAILED SUGGESTIONS FOR MAKING CODE CIF-COMPLIANT")

I. USE THE PROGRAM TEMPLATE

Use the Program Template and its data_manager to carry out all read/write
and parsing of model files. If you do this then all mmCIF/PDB handling is
taken care of for you, except only that you need to capture the actual
file names written by the data_manager.

II. TOOLS ARE AVAILABLE IF YOU CANNOT USE THE PROGRAM TEMPLATE

For existing code that cannot use the Program Template or that
reads and writes files outside of the Program Template:

  a.  Try to pass the data_manager from the Program Template and use
     it to read/write files in your programs. Then once again you only
     need to capture the actual file names written by the data_manager.

  b. If you cannot use the data_manager, use the pdb_or_mmcif_string_info
     method of the model manager or the hierarchy to write your files.
     This method allows setting the preferred output format and capturing
     the name of the actual file that is written.

     Normally you should keep track of the actual file name that is written
     and then use that later when you read the file back in.  If you do
     not use this approach, you can use the get_cif_or_pdb_file_if_present
     tool from iotbx.pdb.utils with your guess of the file name, and it will
     return the name of the file with that name if present, or the name of a
     present file with the opposite extension if it is present instead,
     or blank if neither is present.

     You can use the get_pdb_info tool from iotbx.pdb.utils or the
     iotbx.pdb.input method to read in either mmCIF or PDB formatted files.

III. IF CODE USES PDB-FORMATTED TEXT IT SHOULD BE REWRITTEN

If code parses PDB-formatted text, generally it should be rewritten
to use the methods in the hierarchy class.

  a. For example, code that edits formatted strings representing a PDB file
    to remove HETATM records can be replaced with the method
    remove_hetero() of the hierarchy class.

  b. All code that accumulates lines from multiple PDB files and then interprets
    the new lines should be replaced by reading each PDB file and merging
    the resulting hierarchies or models.  This can be done for model files
    with the simple_combine or add_hierarchies methods available in
    iotbx.pdb.utils, or simple custom code can be written to add chains
    from one hierarchy onto another hierarchy.


IV.  TOOLS ARE AVAILABLE TO FIND CODE THAT NEEDS TO BE MADE CIF-COMPATIBLE
    You can find possibly-problematic code using the tool
    libtbx.find_pdb_mmcif_problems and specifying a file or directory to check

V.  CREATE CIF TESTS TO CHECK CODE WITH MODELS THAT CANNOT FIT IN PDB FORMAT
    For any code that uses pdb/cif files or that uses the hierarchy object
    should be tested with models that do not fit in PDB format.  It is
    recommended that each standard test using models should be duplicated
    and run with non-PDB-compliant models.  The tool
    convert_pdb_to_cif_for_pdb_str can be used to edit strings in place in
    tests so that identical starting strings can be used in the original and
    non-PDB-compliant tests.

VI.  USING FORWARD-COMPATIBLE PDB FORMAT FOR CODE REQUIRING PDB-FORMATTED TEXT
If you have code or 3rd party code that requires PDB-formatted text you can
convert any hierarchy into a forward-compatible hierarchy that can be
formatted in PDB format (hybrid-36 PDB format).  This conversion (currently)
amounts to replacing chainIDs that are longer than 2 characters with
2-character IDs, and residue names that are 5 characters long with
3-character residue names.  A table of conversions is kept that allows
reversion of the hierarchy to its original form.

a.  This procedure is only partially supported and is not encouraged for
   anything except cases that cannot be managed without PDB formatting.

b.  When this is done, it should be carried out either one-way (conversion
    to forward-compatible PDB and never converted back), or else the
    conversion should be carried out, the operation with the converted file
    done, and the result converted back, all in one small block.

c.  Note that conversion may be very complicated if the chain IDs or the
    residue names are needed in whatever operation is done with the
    converted file. This is one of the reasons this approach is not
    recommended except where required. Tools are supplied to convert
    any text-based parameters are used with the converted file, but they
    are not general and not always simple to use.

===========================================================================
===========================================================================

       DETAILED SUGGESTIONS FOR MAKING CODE CIF-COMPLIANT

---------------------------------------------------------------------------
---------------------------------------------------------------------------
I. USING THE PROGRAM TEMPLATE TO HANDLE PDB/MMCIF INPUT/OUTPUT

A. Use the Program Template and use data_manager for
    all model read/write. If you do this there is only one thing
    you need to do:  capture the actual file name written by
    the data_manager.

    The program template automatically adds the scope
    'output.target_output_format' to your parameters, allowing
    a user to set the output format. If it is not set, the program
    template sets it to the format of the default incoming model if
    present, otherwise to 'pdb'.

  def run(self):
    ...
    self.final_file_name = self.data_manager.write_model_file(
        model, self.params.output.file_name)
    print("Final model written to '%s'" %self.final_file_name)

  def get_results(self):
    return group_args(
      output_file_name = self.final_file_name,
      ...)

---------------------------------------------------------------------------
---------------------------------------------------------------------------

II.  TOOLS AVAILABLE IF YOU CANNOT USE THE PROGRAM TEMPLATE

If you do not use the Program template, you will need to keep track
of the target output format and write your files with a pdb/cif-aware
method.

1. Add target_output_format to your "output" or "output_files" scope:

     target_output_format = *None pdb mmcif
       .type = choice
       .help = Desired output format (if possible). Choices are None (\
                try to use input format), pdb, mmcif.  If output model\
                 does not fit in pdb format, mmcif will be used. \
                 Default is pdb.
       .short_caption = Desired output format

2. After you set up your parameters, set the target_output_format:

  from iotbx.pdb.utils import set_target_output_format_in_params
  set_target_output_format_in_params(params)

3. When you write model files, supply the target output format and
  capture the actual file name written:

a. If you have a data manager and a model object:

  file_name = self.data_manager.write_model_file(
        model, self.params.output.file_name)
  print("Model written to '%s'" %file_name)

b. If have only a hierarchy and crystal_symmetry and params:

  info = ph.pdb_or_mmcif_string_info(
           target_format = params.output_files.target_output_format,
           target_filename = params.output.file_name,
           write_file = True,
           crystal_symmetry=crystal_symmetry)
  file_name = info.file_name
  print("Model written to '%s'" %file_name)

c. When you read pdb/mmcif files, if you do not know the ending .pdb or
     .mmcif, use the function get_cif_or_pdb_file_if_present:

   from iotbx.pdb.utils import get_cif_or_pdb_file_if_present
   fn = get_cif_or_pdb_file_if_present(fn)

---------------------------------------------------------------------------
---------------------------------------------------------------------------
III.  REWRITING CODE USING PDB-FORMATTED TEXT

A. You will want to remove all instances of the following methods. None of these
are mmcif-compliant:

  model.model_as_pdb()
  ph.as_pdb_string()
  ph.write_pdb_file()

Suggested replacements:

  1. If your code uses model.model_as_pdb():

    file_name = 'mypdb.pdb'
    str = model.model_as_pdb()
    f = open(file_name,'w')
    print(str, file = f)
    f.close()
    print("Wrote model to '%s'" %file_name)

  Use instead model.pdb_or_mmcif_string_info():

    file_name = 'mypdb.pdb'
    info = model.pdb_or_mmcif_string_info(
        target_format = params.output_files.target_output_format,
        target_filename = file_name,
        write_file = True)
    file_name = info.file_name
    print("Wrote model to '%s'" %file_name)

  2. If your code uses ph.as_pdb_string() and you need the string:

    file_name = 'mypdb.pdb'
    str = ph.as_pdb_string()
    print("Doing something with a string %s" %str)

  Use instead model.pdb_or_mmcif_string_info() which will give you a PDB or
     mmcif string as appropriate:

    info = model.pdb_or_mmcif_string_info(
        target_format = params.output_files.target_output_format)
    str = info.pdb_string
    print("Doing something with a string %s" %str)

  3. If your code uses ph.as_pdb_string() or ph.write_pdb_file() and you are
    writing the string to a file like this:

    file_name = 'mypdb.pdb'
    str = ph.as_pdb_string(crystal_symmetry = crystal_symmetry)
    f = open(file_name,'w')
    print(str, file = f)
    f.close()
    print("Wrote model to '%s'" %file_name)

   or like this:

    file_name = 'mypdb.pdb'
    str = ph.write_model_file(file_name)
    print("Wrote model to '%s'" %file_name)

   Use instead pdb_or_mmcif_string_info:

    file_name = 'mypdb.pdb'
    info = ph.pdb_or_mmcif_string_info(
        target_format = params.output_files.target_output_format,
        target_filename = file_name,
        write_file = True)
    file_name = info.file_name
    print("Wrote model to '%s'" %file_name)

B. You will want to rewrite all code that interprets model text line-by-line.

  If your code looks like this:

  for line in open(model_file).readlines():
    do_something_based_on_a_line_in_file(line)

  You will want instead to read in the model to get a hierarchy, then
  use hierarchy methods to change or interpret the hierarchy.

C. If your code catenates model text like this:

    new_file_name = 'combined.pdb'
    raw_records = list(open(model_file).splitlines())
    raw_records += list(open(other_model_file).splitlines())
    f = open(new_file_name,'w')
    print(raw_records, file = f)
    f.close()
    print("Wrote combined model lines to '%s'" %new_file_name)

   You will want to rewrite it to read in the models and then merge them:

    new_file_name = 'combined.pdb'
    m1 = dm.get_model_file(model_file)
    m2 = dm.get_model_file(other_model_file)
    from iotbx.pdb.utils import simple_combine
    m1 = simple_combine(model_list = [m1,m2]) # NOTE: changes m1 in place
    new_file_name = dm.write_model_file(m1, new_file_name) # capture actual name
    print("Wrote combined model lines to '%s'" %new_file_name)

   Note: you can also merge hierarchies directly with the add_hierarchies method
   in iotbx.pdb.utils

D. If your code names intermediate files with the extension '.pdb':

    new_file_name = 'model.pdb'
    f = open(new_file_name,'w')
    print(model.model_as_pdb(), file = f)
    f.close()
    print("Wrote intermediate model lines to '%s'" %new_file_name)

  Use the data_manager or pdb_or_mmcif_string_info to write the file, and
    capture the actual file name so that you can use it when you read the
    contents of the file back in.

    new_file_name = 'model.pdb'
    new_file_name = dm.write_model_file(model, new_file_name)
    print("Wrote intermediate model lines to '%s'" %new_file_name)


---------------------------------------------------------------------------
---------------------------------------------------------------------------
V.   TOOLS AVAILABLE TO FIND CODE THAT NEEDS TO BE MADE CIF-COMPATIBLE

A.    You can use the libtbx.find_pdb_mmcif_problems to help identify code
   that needs to be modified to make it cif-compatible.

   You can say:

   libtbx.find_pdb_mmcif_problems phenix/phenix/command_line

   and it (recursively) will go through all files/directories in command_line
   and look for problems

   Or you can work on just one file:

   libtbx.find_pdb_mmcif_problems phase_and_build.py

   ====================================================================
   ==>> Here is a **SUPER-HELPFUL HINT** to make the editing easy: <<==
   ====================================================================

B.   Run libtbx.find_pdb_mmcif_problems with the argument "mark_lines":

   1. Run it like this:

      libtbx.find_pdb_mmcif_problems phase_and_build.py mark_lines

   This will edit phase_and_build.py, placing text like:
     " # XXX CHECK PDB: .pdb'"
   at the end of possibly-problematic lines

   2. Then you just edit this file (phase_and_build.py)
     a. find all the places where "CHECK PDB" shows up
     b. fix the problems
     c. mark non-problems with the text " # PDB OK", add explanation if you want

   3. Then clean up by running:

      libtbx.find_pdb_mmcif_problems phase_and_build.py unmark_lines

    which will remove all the " # XXX CHECK PDB" text

   4. Finally, run:

      libtbx.find_pdb_mmcif_problems phase_and_build.py

    again to make sure you got it all.

C. You can run libtbx.find_pdb_mmcif_problems with a file containing a list
  of files:

  1. Put list of your files/directories and put them in files.list:
   files.list:
     phenix/phenix/programs/myfile.py
     phenix/phenix/mydir
   2. Mark/unmark all likely PDB problems in your files
    libtbx.find_pdb_mmcif_problems files.list mark_lines
    libtbx.find_pdb_mmcif_problems files.list unmark_lines

---------------------------------------------------------------------------
---------------------------------------------------------------------------
V.  CREATING CIF TESTS TO CHECK CODE WITH MODELS THAT CANNOT FIT IN PDB FORMAT

  You will want to create a cif-only version of all your tests that use
  models.  The purpose of this is to test all the code that handles
  chain IDs and residue names.

A. Simple conversion of tests to mmCIF if your test uses PDB strings.
   If your test has PDB strings like: pdb_str_1 = """pdb-text""" at the
   top of the file, before any function definitions, you can convert
   the test to mmCIF by placing two lines below all the pdb_str_xxx
   definitions:

   from libtbx.test_utils import convert_pdb_to_cif_for_pdb_str
   convert_pdb_to_cif_for_pdb_str(locals())

  This will make all your PDB strings into mmCIF strings with long
  chain IDS. You can choose how to edit your chain ids if you want:

   convert_pdb_to_cif_for_pdb_str(locals(), chain_addition="ZXLONG")

  You might also want to rename some of your residue names

   You can also do this for strings that have some other name:

   my_string="""text"""
   convert_pdb_to_cif_for_pdb_str(locals(), key_str = 'my_string')

  Now your test should run just as it did before except that the
  chain IDs (and optionally HETATM residue names) may be different.

B. If your test uses models in PDB files, you may simply want to
 make copies of all your models, converting your PDB files into
  mmCIF and edit the chain IDs:

  You can do this with pdbtools:

  phenix.pdbtools x.pdb output.file_name=x.cif old_id=A new_id=AXZLONG

 to make them longer than 2 characters.  If you want, edit
  the HETATM records to change the residue names too.

 Then make a new test tst_xxxx_cif.py that uses these PDB files.
 Note: make sure you rename any restraints file to xxx_restraints.cif
   so you don't overwrite xxx.cif with the new cif-formatted version
   of xxx.pdb

---------------------------------------------------------------------------
---------------------------------------------------------------------------
VI. USING FORWARD-COMPATIBLE PDB FORMAT FOR CODE REQUIRING PDB-FORMATTED TEXT

  The hierarchy class has tools to convert a hierarchy that cannot fit into
   PDB format into one that can and to keep track of the conversion so that
   it can be reversed.

  The main tools are:

  1. Convert a hierarchy to one that fits in PDB format, saving conversion
     information
    fc_ph = ph.as_forward_compatible_hierarchy()

  2. Get the conversion information:
    conversion_info = fc_ph.conversion_info()

  3. Convert back using saved conversion info :
    original_ph = fc_ph.forward_compatible_hierarchy_as_standard()

  4. Convert one-way to PDB format compatible string:
    str =  ph.as_forward_compatible_string()

  5. Identify whether hierarchy has been converted:
    is_fc = ph.is_forward_compatible_hierarchy()

  6. Use saved conversion_info to edit text containing words matching
     chain IDs or residue names, making new text match the chain IDs and
     residue names used in the forward_compatible hierarchy:
    new_text = ph.convert_multi_word_text_to_forward_compatible(text)

  These methods are described in detail in the hierarchy.py code.
---------------------------------------------------------------------------
---------------------------------------------------------------------------
   DETAILS OF METHODS ADDED FOR WORKING WITH PDB/CIF

     MODULE:    iotbx/cli_parser.py

Method to return the parameters as set up by the data_manager:

def get_program_params(run)

     MODULE: iotbx/data_manager/model.py

Method to set the desired output format for model files, based on user
specification and the format of the default input model file:

 def set_target_output_format(self, target_output_format):

Modified write_model_file method to check whether output model fits in PDB
format and to write as mmCIF it does not, or if the user specified cif as the
output format.

 def write_model_file(self, model_str, filename=Auto, extension=Auto,


     MODULE:    iotbx/pdb/hierarchy.py:

Methods for converting hierarchy to forward-compatible (fits in PDB format,
  see above):

  def is_forward_compatible_hierarchy(self):
  def conversion_info(self):
  def convert_multi_word_text_to_forward_compatible(self, text):
  def as_forward_compatible_hierarchy(self, conversion_info = None):
  def forward_compatible_hierarchy_as_standard(self, conversion_info = None):
  def as_forward_compatible_string(self, **kw):

Method for writing as PDB or mmCIF string, using supplied target_format if
possible, optionally writing as file, returning group_args object with
text string, file_name written, and is_mmcif marker:

  def pdb_or_mmcif_string_info(self,

NOTE: This method and the corresponding method in model.py use
default of segid_as_auth_segid = True, different from the default in
model_as_mmcif() in model.py and as_mmcif_string() in hierarchy.py
The default of segid_as_auth_segid = True causes any text in the SEGID
field read from a PDB-formatted file to be written to the auth_segid field
in the mmCIF output, and the chain ID from the PDB file is used as
the actual chain ID in the mmCIF output.

Methods supplied so that code elsewhere does not need to parse PDB formatted
strings to remove HETATM, TER, and BREAK records and to sort chains in order
of chain ID:

  def remove_hetero(self):
  def contains_hetero(self):
  def contains_break_records(self):
  def remove_ter_or_break(self):
  def sort_chains_by_id(self):

     MODULE:    mmtbx/model/model.py:

Method matching one of the same name in hierarchy.py,
for writing as PDB or mmCIF string, using supplied target_format if
possible, optionally writing as file, returning group_args object with
text string, file_name written, and is_mmcif marker:

  def  pdb_or_mmcif_string_info(self,

NOTE: This method and the corresponding method in hierarchy.py use
default of segid_as_auth_segid = True, different from the default in
model_as_mmcif() in model.py and as_mmcif_string() in hierarchy.py.
The default of segid_as_auth_segid = True causes any text in the SEGID
field read from a PDB-formatted file to be written to the auth_segid field
in the mmCIF output, and the chain ID from the PDB file is used as
the actual chain ID in the mmCIF output.

     MODULE:   iotbx/pdb/utils.py:

Methods to set the target_output_format parameter in the scope output/output_files:

def set_target_output_format_in_params(params,
def get_input_model_file_name_from_params(params):
def target_output_format_in_params(params):
def get_target_output_format_from_file_name(file_name,
def move_down_scope_to_input_files(params, levels = 3):

Method to find file named with 'pdb' or 'cif' when given one or the other.
Returns the file supplied if present, otherwise the other file if it is present,
otherwise empty string. Can find files with the pdb/cif followed by an
underscore such as myfile.pdb_1. Only looks for pdb/cif in the extension:

def get_cif_or_pdb_file_if_present(file_name):

Methods to merge hierarchies:

def add_hierarchies(ph_list, create_new_chain_ids_if_necessary = True):
def add_hierarchy(s1_ph, s2_ph, create_new_chain_ids_if_necessary = True):
def catenate_segment_onto_chain(model_chain, s2, gap = 1,
def get_chain(s1_ph, chain_id = None):

Methods to merge and edit models:

def add_model(s1, s2, create_new_chain_ids_if_necessary = True):
def catenate_segments(s1, s2, gap = 1,
def catenate_segment_onto_chain(model_chain, s2, gap = 1,
def simple_combine(model_list,

Method to get hierarchy and pdb_input objects from text files. These differ
from iotbx.pdb.input() and construct_hierarchy() by allowing empty text
for both mmCIF and PDB input, and in packaging a hierarchy, pdb_input, and
crystal_symmetry in a single group_args object that is returned

Normal use:  pdb_info = get_pdb_info(file_name = file_name)

def get_pdb_info(text = None, file_name = None, lines = None,

Shortcuts to get just hierarchy or pdb_input:

def get_pdb_hierarchy(text=None, file_name = None,
def get_pdb_input(text = None, file_name = None, lines = None,

Helper methods for reading pdb_input text and getting hierarchy

def lines_are_really_text(lines):
def get_lines(text = None, file_name = None, lines = None):
def type_of_pdb_input(pdb_inp):
def try_to_get_hierarchy(pdb_inp):

Methods to guess elements in atoms object from a hierarchy:

def check_for_missing_elements(atoms, file_name = None):
def guess_chemical_elements(atoms, check_pseudo = False,
def check_for_incorrect_spacings(atoms):

Method checking for atom names starting with "Z" (used as pseudo-atoms)

def check_for_pseudo_atoms(atoms):


     MODULE:   iotbx/pdb/forward_compatible_pdb_cif_conversion.py

This module has low-level methods for conversion of hierarchies to a PDB-compatible
form, for reading and writing these hierarchies, and for converting them
back to the original form.

Normally the methods in the hierarchy class should be used, rather
than using these low-level methods directly.

The methods available are:

  def __init__(self, hierarchy = None,
  def is_initialized(self):
  def conversion_required(self):
  def conversion_as_remark_hetnam_string(self):
  def convert_hierarchy_to_forward_compatible_pdb_representation(self,
  def convert_hierarchy_to_full_representation(self, hierarchy):
  def set_conversion_tables_from_remark_hetnam_records(
  def _set_up_conversion_table(self, key, hierarchy, unique_values_dict = None):
  def _unique_chain_ids_from_hierarchy(self, hierarchy):
  def _unique_resnames_from_hierarchy(self, hierarchy):
  def _get_any_forward_compatible_pdb_representation(self,
  def _get_forward_compatible_pdb_representation(self, ids, max_chars,
  def _get_new_unique_id(self, id, max_chars, exclude_list,
  def _get_new_id(self, n_chars, exclude_list, end_with_tilde = None,
  def _choose_allowed_ids(self, unique_values, max_chars):
  def _is_allowed(self, u, max_chars):
  def _get_conversion_table_info(self, key):
  def get_full_text_from_forward_compatible_pdb_text(self, key = None,
  def convert_multi_word_text_to_forward_compatible(self,
  def get_forward_compatible_pdb_text_from_full_text(self,


===========================================================================
===========================================================================

'''

def run():
  print(info_string)

if (__name__ == "__main__"):
  run()
