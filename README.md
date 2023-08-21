# movie-asd

## Setup
```
# create a new conda environment
conda create --name movie_asd

# install the requirements
pip install -r requirements.txt

# download the required models
. setup.sh
```
## How to run
```
cd src
python3 main.py --videoPath <path_to_video in mp4> --cacheDir <path to store the intermediate artifacts> --verbose
```


