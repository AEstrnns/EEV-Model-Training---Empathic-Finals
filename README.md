# Contributors (This Repository)
CYS: BCS34
Aira Sophia Estorninos
Joseph Christian Cinco
Ian Carlo Guevarra

# Our Paper Abstract (This Project)
Most conventional surveys that aim to elicit natural subconscious consumer feedback towards video advertisements fail due to self-reporting bias and slogan blindness. To this effect, we assess the ability of deep learning networks to predict continuous subconscious emotion states. Using the Evoked Expressions in Video (EEV) dataset, which includes annotations at a high frequency (6 Hz) of 15 continuous emotions, we tested three models namely CNN, LSTM and CAER-Net-RS. In terms of overall predictive accuracy, LSTM models performed best (MSE = 0.0018, R = 0.9299) and revealed better potential to capture time-varying emotional dynamics. CNN architecture presented very promising spatial feature extraction (MSE = 0.0019), and CAER-Net-RS model showed strong and robust performance of global baseline which implies structural stability in context-based environment data processing. We successfully proved that the spatiotemporal machine learning can measure continuous emotion trajectory in real-time and thus provided an effective, data-driven method for automated audience analysis and neuromarketing assessment. 



# About Evoked Expressions in Video (EEV) Dataset

Videos can evoke a range of affective responses in viewers. The ability to predict evoked affect from a video, before viewers watch the video, can help in content creation and video recommendation. We introduce the Evoked Expressions from Videos (EEV) dataset, a large-scale dataset for studying viewer responses to videos. Each video is annotated at 6 Hz with 15 continuous evoked expression labels, corresponding to the facial expression of viewers who reacted to the video. We use an expression recognition model within our data collection framework to achieve scalability. In total, there are 8 million annotations of viewer facial reactions to 5,153 videos (370 hours). We use a publicly available video platform to obtain a diverse set of video content. We hope that the size and diversity of the EEV dataset will encourage further explorations in video understanding and affective computing.

## Structure of the Dataset

The datset consists of three files, train.csv, val.csv, and test.csv that represent the training, validation, and test splits respectively. We are not releasing the test actual expression scores for the test split at this time but still includes the video IDs and frame timestamps. Each CSV file contains the expected facial expressions of someone reacting to a specific content video. Each line indicates a specific video represented by its video ID, a timestamp in microseconds, and a set of expression scores. The first line of the CSV is a header with the labels for each column. The reaction annotations are sampled at 6 frames per second such that there could be thousands of lines for a single video. Frames or videos where all the predicted expressions have a values of 0.0 represent locations where a detection didn't occur and may be ignored.

## Dataset Over Time

Because this dataset consists fo references to the original source, there may be instances where specific videos are no longer available on the platform and those annotations will be removed from the dataset. This means there is a possibility that the dataset may decrease in size over time.

## Evaluation on the Test Split

Our dataset is used as part of the EEV Challenge at the Affective Understanding in Video (AUVi) Workshop at CVPR 2021. The evaluator on the test split is currently up at: https://www.aicrowd.com/challenges/evoked-expressions-from-videos-challenge-cvpr-2021.

## How to Cite the Dataset

Please consider citing our [paper](https://arxiv.org/abs/2001.05488) if you find the dataset useful:

```
@article{sun2021eev,
      title={EEV: A Large-Scale Dataset for Studying Evoked Expressions from Video}, 
      author={Sun, Jennifer J and Liu, Ting and Cowen, Alan S and Schroff, Florian and Adam, Hartwig and Prasad, Gautam},
      year={2021},
      journal={arXiv preprint arXiv:2001.05488}
}
```
## License

This data is licensed by Google LLC under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/). Users will be allowed to modify and repost it, and we encourage them to analyze and publish research based on the data.

## Contact Us

If you have a technical question regarding the dataset, code or publication, please create an issue in this repository. You may also reach us at eev-dataset@google.com.
