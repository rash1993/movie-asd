# movie-asd

## Setup
```
# Recommended cuda==11.3 (cuda==11.0 also works)

# create a new conda environment
conda create --name movie_asd

# install the requirements using one of the following
# install the requirements (for cuda==11.3)
pip install -r requirements.txt

# install the requriements (for cuda==11.0)
pip install -r requirements_cu_11_0.txt

# download the required models
. setup.sh
```
## How to run
The system runs best for the `*.mp4` formatted videos. 
To use just the unsupervised cross-modal identity association for active speaker detection (CMIA) use the following:
```
cd src
python3 main.py --videoPath <path_to_video in mp4> --cacheDir <path to store the intermediate artifacts> --partitionLength 50 --verbose
```
To run the setup with `TalkNet` as the guide:
```
cd src
python3 main.py --videoPath <path_to_video in mp4> --cacheDir <path to store the intermediate artifacts> --partitionLength 50 --talknet --verbose
```
The above snippet will generate a video with active speakers' faces bounded in a green bounding box while all other boxes are in the red bounding box. An example output video is shown below.

![](https://github.com/rash1993/movie-asd/blob/wacv/gif_v0.gif)

Please cite the following works if you use this framework.
```
@ARTICLE{10102534,
  author={Sharma, Rahul and Narayanan, Shrikanth},
  journal={IEEE Open Journal of Signal Processing}, 
  title={Audio-Visual Activity Guided Cross-Modal Identity Association for Active Speaker Detection}, 
  year={2023},
  volume={4},
  number={},
  pages={225-232},
  doi={10.1109/OJSP.2023.3267269}}
```
```
@article{sharma2022unsupervised,
  title={Unsupervised active speaker detection in media content using cross-modal information},
  author={Sharma, Rahul and Narayanan, Shrikanth},
  journal={arXiv preprint arXiv:2209.11896},
  year={2022}
}
```
