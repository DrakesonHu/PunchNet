PunchNet: Neural Network for Boxing Strike Detection

A computer vision system using transfer learning to classify boxing punches in 
video footage. Built with MATLAB for Brown University's CLPS 0950 (Intro to Programming), Spring 2025.

================================================================================
OVERVIEW
================================================================================

PunchNet analyzes boxing match footage and classifies each punch as a "hit", 
"miss", or "block", providing objective strike counting for amateur boxing events 
where automated scoring systems like CompuBox are unavailable.

================================================================================
ARCHITECTURE
================================================================================

Model: MobileNet-v2 (3.6M parameters)
Input: 224x224x3 RGB video frames
Framework: MATLAB Deep Learning Toolbox
Training: Transfer learning with data augmentation
Dataset: Custom boxing dataset aggregated from Roboflow Universe
(Not included in repository due to size/licensing)

Pipeline:
1. Video Processing - Extract and preprocess frames from video files
2. Frame Classification - MobileNet-v2 classifies each frame independently
3. Temporal Chunking - Group consecutive detections into discrete punch events
4. Strike Counting - Output hit/miss/block statistics

================================================================================
METHODOLOGY
================================================================================

Transfer Learning:
Selected MobileNet-v2 for its lightweight architecture, enabling efficient 
frame-by-frame inference while maintaining classification accuracy. Fine-tuned 
on custom boxing dataset aggregated from Roboflow Universe.

Data Augmentation:
- Random rotation: +/- 20 degrees
- Random translation: +/- 10 pixels (X/Y)
- Random horizontal reflection
Applied during training to improve generalization with limited dataset size.

Hyperparameters:
- Optimizer: SGD with momentum
- Learning rate: Cosine annealing schedule
- L2 regularization: 0.0025
- Batch size: 80
- Epochs: 30
- Validation: 5% split, best-validation model selection

Temporal Chunking:
Implemented difference-based sequence detection to consolidate multi-frame 
punch detections into single discrete events, preventing overcounting.

================================================================================
KEY CHALLENGES
================================================================================

Overfitting Mitigation:
With a limited training dataset, preventing overfitting required extensive 
experimentation with:
- Data augmentation strategies
- Regularization techniques (L2 penalty, dropout)
- Architecture selection (lightweight models reduce overfitting risk)
- Early stopping via validation monitoring

Frame Chunking Logic:
Raw frame-by-frame classification detects individual punches across multiple 
consecutive frames. Developed custom temporal clustering to merge these into 
accurate punch counts.

Temporal Limitations:
This frame-by-frame approach fundamentally limits accuracy because each frame 
is processed independently without motion context. Accurate punch detection 
requires temporal attention mechanisms to model motion across frames - e.g., recurrent architectures (ConvLSTM), or transformer-based video models with temporal self-attention. 
Our post-hoc chunking heuristic partially addresses this but cannot capture 
true motion dynamics. Frame-by-frame, a video may not capture the frame of connection. Additionally, there may be blurring and ambiguity that can only be resolved with motion data. Therefore, our implementation is fundamentally underfitted for the task. Due to time and data limitations, we settled on our approach as a proof-of-concept.

================================================================================
RESULTS
================================================================================

Training accuracy: Monitored across 30 epochs with cosine learning rate decay
Validation performance: Model selection based on best validation loss
Outputs: Per-class precision and confusion matrix for hit/miss/block classification

See training.png for training progression visualization.

================================================================================
PROJECT STRUCTURE
================================================================================

Master.m              - Main execution script
TransferNew.m         - Transfer learning training pipeline
Video_Process.m       - Video frame extraction
Chunk.m               - Temporal punch detection chunking
net_1.mat             - Trained MobileNet-v2 weights
PDAugmented2/         - Training dataset (organized by class)
Video_Data/           - Test video files
training.png          - Training loss/accuracy visualization

================================================================================
TEAM CONTRIBUTIONS
================================================================================

Drakeson Hu: ML architecture, transfer learning implementation, hyperparameter 
tuning, dataset preparation, overfitting mitigation strategies

Frances Robertson: Temporal chunking algorithm, frame sequence processing, 
visualization

Donavan Jeng: Video processing pipeline, frame extraction, file I/O handling

================================================================================
LIMITATIONS & FUTURE WORK
================================================================================

Current Limitations:
- Frame-by-frame processing lacks temporal context
- No attention mechanism for motion modeling
- Analyzes uploaded video files (not live)
- Counts total strikes without distinguishing between fighters
- No user interface

Future Enhancements:
- Implement video network architectures (3D CNN, ConvLSTM, or temporal 
  transformers) for true motion understanding
- Fighter-specific punch tracking via pose estimation
- Real-time processing for live match scoring
- Integration with existing boxing analytics platforms

================================================================================
SETUP
================================================================================

Trained model weights are available in GitHub Releases:
https://github.com/DrakesonHu/PunchNet/releases

Download net_1.mat and place in the root directory before running Master.m

================================================================================

Course: CLPS 0950 Intro to Programming, Brown University, Spring 2025
Language: MATLAB
Architecture: MobileNet-v2 with transfer learning