"""Converting markdown files into jupyter notebooks
"""
import notedown
import glob
import pkg_resources
import nbformat
import re
import shutil

class Markdown2Notebook:
    def __init__(self):
        # timeout in sec to evaluate each notebook
        self.timeout = 200
        # limit the number of lines in a cell output
        self.max_output_length = 1000
        # a list of converted files, format is (old_file_name, new_file_name)
        self.converted_files = []
        # the list of notebooks to skip evaluation
        self.skip_evaluation = []


        # parse if each markdown file is actually a jupyter notebook
        valid_notebooks = []
        for fname in glob.glob('*.md'):
            with open(fname, 'r') as fp:
                valid_notebooks.append(
                    (fname, '```{.python .input' in fp.read()))
        self.valid_notebooks = dict(valid_notebooks)
        print(self.valid_notebooks)

    def _remove_chaper(self, fname):
        """remove the heading P01-C02"""
        parts = fname.split('-')
        if len(parts) < 3:
            return fname
        if ((parts[0].startswith('A') or parts[0].startswith('P'))
            and parts[1].startswith('C')):
            return '-'.join(parts[2:])
        return fname

    def _get_new_fname(self, fname):
        """C01-P01-haha.md -> haha.ipynb"""
        new_fname = self._remove_chaper(fname)
        if new_fname == fname:
            return fname
        parts = new_fname.split('.')
        if self.valid_notebooks[fname]:
            parts[-1] = 'ipynb'
        else:
            assert parts[-1] == 'md'
        return '.'.join(parts)

    def update_links(self, content):
        """Update all C01-P01-haha.md into haha.ipynb"""
        link_md = re.compile('.*\]\(([\w/.-]*)\)')
        link_rst = re.compile('.*\<([\w/.-]*)\>`\_')
        lines = content.split('\n')
        for i,l in enumerate(lines):
            m = link_md.match(l) or link_rst.match(l)
            if not m:
                continue
            for link in m.groups():
                if not link.endswith('.md'):
                    continue
                new_link = self._remove_chaper(link)
                if new_link != link:
                    lines[i] = l.replace(link, new_link.replace('.md', '.html'))
                    print(lines[i])
        return '\n'.join(lines)

    def _has_output(self, notebook):
        """return if a notebook contains at least one output cell"""
        return len(self._get_outputs(notebook))

    def _get_outputs(self, notebook):
        """get all output cells"""
        outs = []
        for cell in notebook.cells:
            if 'outputs' in cell:
                src = cell['source']
                for o in cell['outputs']:
                    if 'text' in o:
                        outs.append((src, o['text']))
        return outs

    def _output_sanity_check(self, text):
        """check the cell outputs"""
        lines = text.split('\n')
        assert len(lines) < self.max_output_length, 'Too long cell output'
        for l in lines:
            assert not ('Error' in l and 'Traceback' in l), 'Cell output contains errors'

    def convert(self):
        """Find all markdown files, convert into jupyter notebooks
        """
        reader = notedown.MarkdownReader()
        writer = nbformat

        for fname in glob.glob('*.md'):
            new_fname = self._get_new_fname(fname)
            if new_fname == fname:
                print('=== skipping %s' % (fname))
                continue

            if not self.valid_notebooks[fname]:
                print('=== renaming %s to %s' % (fname, new_fname))
                shutil.copyfile(fname, new_fname)
            else:
                print('=== converting %s to %s' % (fname, new_fname))
                # read
                with open(fname, 'r') as fp:
                    notebook = reader.read(fp)

                # evaluate notebook
                if not (self._has_output(notebook) or any(
                        [f in fname for f in self.skip_evaluation])):
                    print('Evaluating %s with timeout %d sec' % (fname, self.timeout))
                    notedown.run(notebook, self.timeout)

                # check output
                for src, output in self._get_outputs(notebook):
                    try:
                        self._output_sanity_check(output)
                    except AssertionError:
                        print('Output error for the cell starting with:')
                        print(src.split('\n')[0:3])
                        raise

                # TODO(mli) pylint check

                # write
                with open(new_fname, 'w') as f:
                    f.write(writer.writes(notebook))

            self.converted_files.append((fname, new_fname))
