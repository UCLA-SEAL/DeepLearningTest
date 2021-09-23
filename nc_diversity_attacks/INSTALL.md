# INSTALL

The following steps should be sufficient to get these attacks up and running on most systems running Python 3.7.3+.

1. Download a suitable version of [PyTorch](https://pytorch.org/get-started/locally/) for your environment. 
	- This project originally ran with `torch==1.5.0` and `torchvision==0.6.0`, but other versions are likely to work as well.
	- It is also *highly* recommended that you use GPUs to execute the evaluation scripts.

2. Run the `requirements.txt` file:
```
numpy       ~=1.16.2
pandas      ~=0.24.2
tqdm        ~=4.31.1
matplotlib  ~=3.0.3
scipy       ~=1.2.1
seaborn     ~=0.9.0
```
Note: these are the most recent versions of each library used, lower versions may be acceptable as well. If you have other versions Python, the `~=` operator should download the best compatible version of each package for you.  

```
pip install -r requirements.txt
```

3. Viewing results in Jupyter Notebooks

The results are aggregated and visualized in a `jupyter notebook`, which can be viewed directly in GitHub or perused locally:
```
# install
pip install jupyter

# start notebook in working directory
jupyter notebook
```