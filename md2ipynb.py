"""Converting markdown files into jupyter notebooks
"""
import notedown
import glob
import pkg_resources
import nbformat
import re

class Markdown2Notebook:
    def __init__(self):
        self.timeout = 10
        self.max_output_length = 1000
        self.converted_files = []

    def _startswith_chapter(self, fname):
        """return if fname starts with P01-C02"""
        return self._remove_chaper(fname) != fname

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
        if parts[-1] == 'md':
            parts[-1] = 'ipynb'
        return '.'.join(parts)

    def update_links(self, content):
        """Update all C01-P01-haha.md into haha.ipynb"""
        link_re = re.compile('.*\]\(([\w/.-]*)\)')
        lines = content.split('\n')
        for i,l in enumerate(lines):
            m = link_re.match(l)
            if not m:
                continue
            for link in m.groups():
                new_link = self._get_new_fname(link)
                if new_link != link:
                    lines[i].replace(link, new_link)
        return '\n'.join(lines)

    def _update_links(self, notebook):
        """update the links in notebook"""
        num = 0
        link_re = re.compile('.*\]\(([\w/.-]*)\)')
        for cell in notebook['cells']:
            if cell['cell_type'] != 'markdown':
                continue
            src = cell['source'].split('\n')
            for i,l in enumerate(src):
                m = link_re.match(l)
                if not m:
                    continue
                for link in m.groups():
                    new_link = self._get_new_fname(link)
                    if new_link != link:
                        src[i].replace(link, new_link)
                        num += 1
        return num

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

    def _valid_notebook(self, notebook):
        """return if this is a jupyter notebook, not a pure markdown file"""
        return len(notebook.cells) > 1

    def convert(self):
        """Find all markdown files, convert into jupyter notebooks
        """
        reader = notedown.MarkdownReader()
        template_file = pkg_resources.resource_filename(
            'notedown', 'templates/markdown.tpl')
        writer = {'ipynb':nbformat,
                  'md':notedown.MarkdownWriter(template_file, output_dir='./')}

        for fname in glob.glob('*.md'):
            new_fname = self._get_new_fname(fname)
            print(new_fname)
            if new_fname == fname:
                continue
            print('=== converting %s to %s' % (fname, new_fname))

            # read
            with open(fname, 'r') as fp:
                notebook = reader.read(fp)

            # update link
            for cell in notebook['cells']:
                if cell['cell_type'] == 'markdown':
                    cell['source'] = self.update_links(cell['source'])

            if self._valid_notebook(notebook):
                # evaluate notebook
                if not self._has_output(notebook):
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
            ext = new_fname.split('.')[-1]
            output = writer[ext].writes(notebook)
            with open(new_fname, 'w') as f:
                f.write(output)

            self.converted_files.append((fname, new_fname))
