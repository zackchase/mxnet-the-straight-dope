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

def _replace_ext(fname, new_ext):
    """replace the file extension in a filename"""
    parts = fname.split('.')
    if len(parts) <= 1:
        return fname
    parts[-1] = new_ext
    return '.'.join(parts)

class RenameFiles:
    def __init__(self):
        self.renamed_files = []
        self.header_re = re.compile("([PA][\d\.]+-C[\d\.]+-)(.*)")

    def _get_new_fname(self, fname):
        """P01-C01-haha.ipynb -> haha.ipynb"""
        m = self.header_re.match(fname)
        return m.groups()[1] if m else fname

    def rename(self):
        """remove all md and ipynb files"""
        for fname in glob.glob('*.md') + glob.glob('*.ipynb'):
            new_fname = self._get_new_fname(fname)
            if fname != new_fname:
                print('=== rename %s to %s' % (fname, new_fname))
                shutil.copyfile(fname, new_fname)
                self.renamed_files.append((fname, new_fname))

    def update_links(self, app, docname, source):
        """Update all C01-P01-haha.md into haha.html"""
        def _new_url(m):
            assert len(m.groups()) == 1, m
            url = m.groups()[0]
            if url.startswith('./'):
                url = url[2:]
            if self._get_new_fname(url) != url:
                url = _replace_ext(self._get_new_fname(url), 'html')
            return url

        for i,j in enumerate(source):
            if os.path.exists(docname+'.md') or os.path.exists(docname+'.ipynb'):
                source[i] = re.sub('\]\(([\w/.-]*)\)',
                                   lambda m : ']('+_new_url(m)+')', j)
            elif os.path.exists(docname+'.rst'):
                source[i] = re.sub('\<([\w/.-]*)\>`\_',
                                   lambda m: '<'+_new_url(m)+'>`_', j)

class Markdown2Notebook:
    def __init__(self, ignore_list):
        # the files to be ignored for converting
        self.ignore_list = ignore_list
        # timeout in second to evaluate a notebook
        self.timeout = 400
        # a list of converted files, format is (old_file_name, new_file_name)
        self.converted_files = []

    def _has_output(self, notebook):
        for cell in notebook.cells:
            if 'outputs' in cell and cell['outputs']:
                return True
        return False

    def convert(self):
        """Find all markdown files, convert into jupyter notebooks
        """
        reader = notedown.MarkdownReader()
        files = [fname for fname in glob.glob('*.md') if fname not in self.ignore_list]


        valid_notebooks = []
        for fname in files:
            with open(fname, 'r') as fp:
                valid_notebooks.append(
                    (fname, '```{.python .input' in fp.read()))
        self.valid_notebooks = dict(valid_notebooks)

        for fname in files:
            # parse if each markdown file is actually a jupyter notebook
            with open(fname, 'r') as fp:
                valid = '```{.python .input' in fp.read()
                if not valid:
                    print('=== skip convert %s' % (fname))
                    continue

            # read
            with open(fname, 'r') as f:
                notebook = reader.read(f)

            if not self._has_output(notebook):
                print('=== evaluate %s with timeout %d sec'%(fname, self.timeout))
                tic = time.time()
                notedown.run(notebook, self.timeout)
                print('=== finished in %f sec'%(time.time()-tic))

            # write
            new_fname = _replace_ext(fname, 'ipynb')
            print('=== convert %s to %s' % (fname, new_fname))
            with open(new_fname, 'w') as f:
                f.write(nbformat.writes(notebook))

            self.converted_files.append((fname, new_fname))

class SanityCheck:
    def __init__(self, ignore_list):
        # limit the number of lines in a cell output
        self.max_output_length = 1000
        self.ignore_list = ignore_list

    def _check_cell_output(self, text):
        """check the cell outputs"""

    def _check_notebook(self, fname):
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
                     assert nlines < self.max_output_length, 'Too long cell output'
                 except AssertionError:
                     print('This cell\'s output contains error:\n')
                     print('-'*40)
                     print(src)
                     print('-'*40)
                     raise

    def check(self, app, exception):
        for fname in glob.glob('*.ipynb'):
            if fname in self.ignore_list:
                continue
            print('=== check '+fname)
            self._check_notebook(fname)


renamer = RenameFiles()
renamer.rename()
ignore_list = [f for f,_ in renamer.renamed_files]

convert = Markdown2Notebook(ignore_list=ignore_list)
convert.convert()
ignore_list += [f for f, _ in convert.converted_files]

checker = SanityCheck(ignore_list=ignore_list)

def remove_generated_files(app, exception):
    for _, f in renamer.renamed_files + convert.converted_files:
        print('=== remove %s' % (f))
        os.remove(f)
