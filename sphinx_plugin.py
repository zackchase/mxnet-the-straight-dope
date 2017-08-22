"""
1. Converting markdown files into jupyter notebooks
2. Remove filename headers, such as from P01-C01-xx.ipynb to xx.ipynb
"""
import notedown
import glob
import pkg_resources
import nbformat
import re
import shutil
import os
import time

# timeout in second to evaluate a notebook
timeout = 400
# limit the number of lines in a cell output
max_output_length = 1000
# the files will be ingored for execution
ignore_execution = ['P01', 'P02']

def _replace_ext(fname, new_ext):
    """replace the file extension in a filename"""
    parts = fname.split('.')
    if len(parts) <= 1:
        return fname
    parts[-1] = new_ext
    return '.'.join(parts)

def _get_new_fname(fname):
    """P01-C01-haha.ipynb -> haha.ipynb"""
    header_re = re.compile("([PA][\d\.]+-C[\d\.]+-)(.*)")
    m = header_re.match(fname)
    return m.groups()[1] if m else fname

def _has_output(notebook):
    """if a notebook contains output"""
    for cell in notebook.cells:
        if 'outputs' in cell and cell['outputs']:
            return True
    return False

def convert_md():
    """Find all markdown files, convert into jupyter notebooks
    """
    converted_files = []
    reader = notedown.MarkdownReader()
    for fname in glob.glob('*.md'):
        new_fname = _get_new_fname(fname)
        # parse if each markdown file is actually a jupyter notebook
        with open(fname, 'r') as fp:
            valid = '```{.python .input' in fp.read()
            if not valid:
                if new_fname != fname:
                    print('=== Rename %s -> %s' % (fname, new_fname))
                    shutil.copyfile(fname, new_fname)
                    converted_files.append((fname, new_fname))
                continue

        # read
        with open(fname, 'r') as f:
            notebook = reader.read(f)

        if not (_has_output(notebook) or
                any([i in fname for i in ignore_execution])):
            print('=== Evaluate %s with timeout %d sec'%(fname, timeout))
            tic = time.time()
            notedown.run(notebook, timeout)
            print('=== Finished in %f sec'%(time.time()-tic))

        # write
        new_fname = _replace_ext(new_fname, 'ipynb')
        print('=== Convert %s -> %s' % (fname, new_fname))
        with open(new_fname, 'w') as f:
            f.write(nbformat.writes(notebook))

        converted_files.append((fname, new_fname))
    return converted_files

def rename_ipynb():
    """renmae all ipynb files"""
    renamed_files = []
    for fname in glob.glob('*.ipynb'):
        new_fname = _get_new_fname(fname)
        if fname != new_fname:
            print('=== Rename %s -> %s' % (fname, new_fname))
            shutil.copyfile(fname, new_fname)
            renamed_files.append((fname, new_fname))
    return renamed_files

def update_links(app, docname, source):
    """Update all C01-P01-haha.md into haha.html"""
    def _new_url(m):
        assert len(m.groups()) == 1, m
        url = m.groups()[0]
        if url.startswith('./'):
            url = url[2:]
        if _get_new_fname(url) != url:
            url = _replace_ext(_get_new_fname(url), 'html')
        return url

    for i,j in enumerate(source):
        if os.path.exists(docname+'.md') or os.path.exists(docname+'.ipynb'):
            source[i] = re.sub('\]\(([\w/.-]*)\)',
                               lambda m : ']('+_new_url(m)+')', j)
        elif os.path.exists(docname+'.rst'):
            source[i] = re.sub('\<([\w/.-]*)\>`\_',
                                   lambda m: '<'+_new_url(m)+'>`_', j)

def check_output(app, exception):
    for fname in glob.glob('*.ipynb'):
        print('=== Check '+fname)

        with open(fname, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        # TODO(mli) lint check
        for cell in nb.cells:
             if 'outputs' in cell:
                 src = cell['source']
                 nlines = 0
                 try:
                     for o in cell['outputs']:
                         if 'text' in o:
                             nlines += len(o['text'].split('\n'))
                         assert 'traceback' not in o, '%s, %s'%(o['ename'], o['evalue'])
                     assert nlines < max_output_length, 'Too long cell output'
                 except AssertionError:
                     print('This cell\'s output contains error:\n')
                     print('-'*40)
                     print(src)
                     print('-'*40)
                     raise

converted_files = convert_md()
renamed_files = rename_ipynb()
ignore_list = [f for f,_ in converted_files + renamed_files]
print(ignore_list)

def remove_generated_files(app, exception):
    for _, f in renamed_files + converted_files:
        print('=== Remove %s' % (f))
        os.remove(f)
