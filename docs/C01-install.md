# Run these tutorials

Each tutorial consists of a Jupyter notebook, which is editable and
runnable. To run these notebooks, you must have `python` installed.
Additionally, you'll need `jupyter` and a recent version of `mxnet`.  
The following commands install them through `pip`:

```bash
# optional: update pip to the newest version
sudo pip install --upgrade pip
# install jupyter
pip install jupyter --user
# install the nightly built mxnet
pip install mxnet --pre --user
```

The default `MXNet` package only supports CPU but some tutorials require
GPUs. If you are running on a computer that has a GPU and either CUDA 7.5 
or 8.0 is installed, then the following commands install a GPU-enabled 
version of MXNet. 

```bash
pip install mxnet-cu80 --pre --user  # for CUDA 8.0
pip install mxnet-cu90 --pre --user  # for CUDA 9.0
```

After completing installation, you're ready to obtain and run the source code:

```bash
git clone https://github.com/zackchase/mxnet-the-straight-dope/
cd mxnet-the-straight-dope
jupyter notebook 
```

The last command starts the Jupyter notebook. You can now run and edit the 
notebooks in a web browser. If you're running the notebooks on a server,
then you might want to ssh with the `-L` flag to tie localhost:8888
on your machine and on the server:

```
ssh myserver -L 8888:localhost:8888
```

Pro tip: if you'd like to run your notebook on some other port (than 8888),
launch it with:

```jupyter notebook --port <port_number>```
