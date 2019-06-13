# Home Grasping Dataset

Tools for downloading and using the home grasping dataset in the [Robot Learning in Homes](http://papers.nips.cc/paper/8123-robot-learning-in-homes-improving-generalization-and-reducing-dataset-bias.pdf) paper. In this pruned dataset, there are 28,874 grasp attempts with a 18.6% grasp success rate. 

## Download links
There are three datasets you can work with from this repository:
* [The full raw grasping data collected on our low cost robot (~30GB)](https://www.dropbox.com/s/njw0extmmon2yro/grasping_data.tar.gz)
* [A smaller subsample of the raw grasping data collected on our low cost robot (~1.2GB)](https://www.dropbox.com/s/vzricn40z2n4la5/grasping_data_small.tar.gz)
* [The patch dataset extracted from the full grasp dataset (~5.3GB)](https://www.dropbox.com/s/k5eogg3nuc5ybtv/patch_dataset.tar.gz)

## Data organization
The raw grasping data is organized as follows:

```bash
.
├── grasping_data
│   ├── <ROBOT_ID>
│   │   ├── <GRASP_ENVIRONMENT>
│   │   │   ├── grasp_<ATTEMPT_ID>
│   │   │   │   ├── color.jpg
│   │   │   │   ├── depth.jpg
│   │   │   │   ├── data.p
└── .   .   .   .
```

`ROBOT_ID` runs from 1 to 5 and corresponds to the robot on which the data was collected. `GRASP_ENVIRONMENT` corresponds to a grasping run. All the grasps in a specific environment run are in the same home. `ATTEMPT_ID` corresponds to a specific grasp attempt in the `GRASP_ENVIRONMENT`. Each grasp contains the RGB `color.jpg` image, the Depth `depth.jpg` image, and a pickle file `data.p` that contains information about the grasp attempt (like success or failure).

## Getting started with the data
To get started with this data, we will use the smaller subsample. To get the full data, change the `wget` download to https://www.dropbox.com/s/njw0extmmon2yro/grasping_data.tar.gz.

```bash
cd ~
git clone https://github.com/lerrel/home_dataset.git
cd home_dataset
wget https://www.dropbox.com/s/vzricn40z2n4la5/grasping_data_small.tar.gz
tar -xvzf grasping_data_small.tar.gz
```

Run the following script in `python 2.7` to display and extract the patch dataset from the raw grasping data.

```bash
python scripts/extract_patch_dataset.py --home_dataset_path ~/home_dataset/grasping_data_small --patch_dataset_path '/tmp/' --train_fraction 0.8 --display 1 --msec 1000
```

To download pre-extracted patch dataset:

```bash
cd ~/home_dataset
wget https://www.dropbox.com/s/k5eogg3nuc5ybtv/patch_dataset.tar.gz
tar -xvzf Patch_Dataset.tar.gz
```

To visualize the patch dataset, run:

```bash
python scripts/display_patch_dataset.py --patch_dataset_path ~/home_dataset/patch_dataset/Train --pos 1 --rand 0 --msec 1000
```

## Acknowledgments

The grasp data used is from the [Robot Learning in Homes](http://papers.nips.cc/paper/8123-robot-learning-in-homes-improving-generalization-and-reducing-dataset-bias.pdf) paper.

```
@inproceedings{gupta2018robot,
  title={Robot learning in homes: Improving generalization and reducing dataset bias},
  author={Gupta, Abhinav and Murali, Adithyavairavan and Gandhi, Dhiraj Prakashchand and Pinto, Lerrel},
  booktitle={Advances in Neural Information Processing Systems},
  pages={9112--9122},
  year={2018}
}
```
