# Audio Feature Extraction
The aim of this project is the discover what combination of audio features gives the best performance with electronic versus organic source classification. Source recognition is treated as a binary classification problem, with a sound represented as either orginating from a live in-person source or an electronic source. Features extracted by [Essentia](http://essentia.upf.edu/) and [LibROSA](https://librosa.github.io/librosa/), tools for audio analysis and audio-based music information retrieval, were used.

### Notebooks
- [audio-ml-extraction.ipynb](https://github.com/rramnauth2220/audio-feature-extraction/blob/master/audio-ml-extraction.ipynb) contains ML classifiers implemented on existing dataset
- [essentia_all_features.ipynb](https://github.com/rramnauth2220/audio-feature-extraction/blob/master/essentia_all_features.ipynb) contains methods for extracting relevant audio features using Essentia
- [librosa_feature_extraction.ipynb](https://github.com/rramnauth2220/audio-feature-extraction/blob/master/librosa_feature_extraction.ipynb) contains methods for extracting relevant audio features using LibROSA

### Outputs
- [librosa_features.csv](https://github.com/rramnauth2220/audio-feature-extraction/blob/master/librosa_features.csv) = example output of LibROSA extraction
- [features_mfccaggr.sig](https://github.com/rramnauth2220/audio-feature-extraction/blob/master/features_mfccaggr.sig) = example output of signal-level Essentia extraction
- [features_frames_aggr.sig](https://github.com/rramnauth2220/audio-feature-extraction/blob/master/features_frames_aggr.sig) = example output of frame-level Essentia extraction

### Directories
- [/audio](https://github.com/rramnauth2220/audio-feature-extraction/tree/master/audio) = current audio dataset
- [/tutorials](https://github.com/rramnauth2220/audio-feature-extraction/tree/master/tutorials) = working Essentia examples
