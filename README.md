# Reachability in Pytorch

This is a minimal repository for doing HJ reachability analysis using the Discounted Safety/Reach-avoid Bellman equation originally introduced in [this](https://ieeexplore.ieee.org/abstract/document/8794107) and [this](https://arxiv.org/abs/2112.12288) paper. We build on the implementation used as baselines from [Jingqi Li's Repo](https://github.com/jamesjingqili/Lipschitz_Continuous_Reachability_Learning). 


This branch of the repository supports a DDPG implementations of the safety-only value functions. I have not yet tested HJ with disturbances.


We recommend Python version 3.12. 

Install instruction:

1. git clone the repo

2. cd to the root location of this repo, where you should be able to see the "setup.py". Note that if you use MacOS, then pytorch 2.4.0 is not available, and therefore you have to first change the line 22 of setup.py from "pytorch==2.4.0" to "pytorch==2.2.2", and then do the step 3. (However, Pytorch==2.4.0 is available for Ubuntu systems. So, if you use Ubuntu, then you can directly go to step 3. )

3. run in terminal: pip install -e .

4. run in terminal: conda install -c conda-forge ffmpeg


# Some sample training scripts:


## DINO-WM Training

To train DINO-WM, we need to first collect all of our data. The repository assumes data is collected in a folder `/data` with trajectories by the name `traj_XXXX.hdf5`. We provide one example of such a trajectory in `dino_wm/traj_0000.hdf5`.

First, we label each trajectory with failure labels. We provide a script to do so, but feel free to devise other ways beyond hand-labeling for this step.
> cd dino_wm && python label.py

We now consolidate all of our data into one file using
> python hdf5_to_dataset.py

This may take a while. We recommend setting aside some trajectories for evaluation in a separate directory (e.g., `/data-test`) and rerunning this script.

We can now proceed to train DINO-WM. First, train the decoder:
> python train_dino_decoder.py

With the decoder in hand, we can now qualitatively monitor the world model's training:
> python train_dino_wm.py

Finally, we train the failure classifier with 
> python train_dino_classifer.py

We can see how well our failure classifier does by computing a confusion matrix for how well our failure classifer performs on the held-out data.
> python eval_dino_classifier.py

If all looks good (low FP and FN rates), lets train our latent safety filter!

> cd ../scripts && python run_training_ddpg-dinowm.py


We can similarly evaluate our BRT using 
> cd ../dino_wm && python eval_dino_brt.py


## Helpful tips:
You may want to crop your front-facing camera view to be a bit more in focus. We provide `vis_data.ipynb` to let you view your camera data and the corresponding DINOv2 embeddings.
