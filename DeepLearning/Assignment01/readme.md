# Deep Learning Assignment 01

## Files Structure

```
Assignment01/
├── question1.py          # Python script for Question 1
├── question1.ipynb        # Jupyter notebook for Question 1
├── question2.py           # Python script for Question 2
├── question2.ipynb        # Jupyter notebook for Question 2
├── requirements.txt       # Required dependencies
└── readme.md             # This file
```

## PC-Configuration

- OS: Windows 11
- Processor: AMD Ryzen 7 5700X
- RAM: 32 GB
- Graphic: AMD Radeon RX 6650 XT (Non-Nvidia)

## Prerequisites

I used Python 3.11, So Please make sure you have Python 3.11 installed.

## Setup Instructions

1. **Extract Zip file and Create and activate a virtual environment**

   ```bash
   cd \Assignment01
   # Create virtual environment
   python3 -m venv venv
   # Activate on Windows
   venv\Scripts\activate
   # Activate on Linux/Mac
   source venv/bin/activate
   ```

2. **Install required dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## How to Execute

**Question 1:** Reduce `epochs` for faster output. I used `epochs=20` but it took 1.5 hour to train both models. My PC has AMD GPU which doesn't support CUDA cores.

```bash
python question1.py
```

**Question 2:**

```bash
python question2.py
```

## Expected Output

### Question 1

- **Output**:
  - Console output showing model training progress
  - Performance metrics (accuracy, loss, etc.)
  - Visualization plots will be available in `/output/q1` folder
  - Model evaluation results

### Question 2

- **Output**:
  - Training/validation curves
  - Model predictions
  - Performance comparisons
  - Generated plots and charts

## Performance Notes

- **PC**: with above configuration described above it took 1.5 hours-2 hours to complete training.
- **Google Colab**: I used Google Colab and it was faster, but I was getting blank images while reconstructing images from Contractive Auto Encoders where blank.
- **Azure AI Studio**: I used Azure AI Studio and it was slower, because they were not providing GPU with student or Visual Studio Subscription.
- **kaggle** kaggle has same issue. They are also not providing GPUs by default. found my PC faster better in this case.
