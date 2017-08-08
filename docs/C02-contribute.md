# How to contribute

For whinges and inquiries, please open
[an issue at github](https://github.com/zackchase/mxnet-the-straight-dope/issues).

To contribute codes, please follow the following guidelines:

1. Check the
   [roadmap](https://github.com/zackchase/mxnet-the-straight-dope/#roadmap)
   before creating a new tutorial.

2. Only cover a single new concept on a tutorial, and explain it in detail. Do
   not assume readers will know it before.

3. Make both words and codes as simple as possible. Each tutorial should take
   no more than 20 minutes to read

4. Do not submit large files, such as dataset or images, to the repo. You can
   upload them to a different repo and cross reference it. For example

   - Insert an image:

     ```
     ![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mnist.png)
     ```

   - Download a dataset if not exists in local:

     ```
     mx.test_utils.download('https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/ptb/ptb.train.txt')
     ```

5. Resize the images to proper sizes. Large size images look fine in notebook,
   but they may be ugly in the HTML or PDF format.

6. Either restart and evaluate all code blocks or clean all outputs before
   submitting

   - For the former, you can click `Kernel -> Restart & Run All` in the
     Jupyter notebook menu.
   - For the latter, use `Kernel -> Restart & Clear Output`. Then our Jenkins
     server will evaluate this notebook when building the documents. It is
     recommended because it can be used as a unit test. But only do it if this
     notebook is fast to run (e.g. less than 5 minutes) and does not require
     GPU.

7. You can build the documents locally to preview the changes. It requires GPU
   is available with `CUDA 8.0` installed, and also `conda` is installed.  T
   following commands create an environment with all requirements installed:

    ```bash
    # assume at the root directory of this project
    conda env create -f environment.yml
    source activate gluon_docs
    ```

   Now you are able to build the HTMLs::

    ```bash
    make html
    ```
