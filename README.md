# movie-asd

Hi there! This repo provides the code and setup for 
* [Cross-modal identity association (CMIA)](https://arxiv.org/abs/2209.11896) framework for active speaker detection. 
* [Audio-visual activity guided CMIA for active speaker detection](https://ieeexplore.ieee.org/abstract/document/10102534).

This setup use the [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD) as the source for audio-visual activity information. 

For a quick read on the technical aspects and video illustrations refer to the [blog on medium](https://medium.com/@rahulsharma-sail/unveiling-the-future-of-media-analysis-revolutionizing-active-speaker-detection-in-the-digital-age-a63316ebae42?sk=d4fd94a03593ad2613f2591fb6106c0a). For queries regarding the setup reach out to [rahul.sharma@usc.edu](mailto:rahul.sharma@usc.edu)

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
To use the unsupervised cross-modal identitiy association for active speaker detection (CMIA) use the following:
```
cd src
python3 main.py --videoPath <path_to_video in mp4> --cacheDir <path to store the intermediate artifacts> --partitionLength 50 --verbose
```
To run the setup with audio-visual activity information from `TalkNet` as the guides for CMIA:
```
cd src
python3 main.py --videoPath <path_to_video in mp4> --cacheDir <path to store the intermediate artifacts> --partitionLength 50 --talknet --verbose
```
The above snippet will generate a video with active speakers' faces bounded in a green bounding box while all other boxes are in the red bounding box. An example output video is shown below.

![](https://github.com/rash1993/movie-asd/blob/wacv/gif_v0.gif)

The improved performance with the use of `TalkNet` comes with increased processing time. In case of smaller videos (<5min) removing the field `--partitionLength` may improve performance with a slight increase in processing time. For the longer videos the `--partitionLength` is important for reasonable processing time and we recommend keeping it `50` is recommended.

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
