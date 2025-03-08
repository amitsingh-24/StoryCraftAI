# StoryCraftAI

[![CI/CD Deployment](https://github.com/yourusername/StoryCraftAI/actions/workflows/cd.yml/badge.svg)](https://github.com/amitsingh-24/StoryCraftAI/blob/main/.github/workflows/deploy.yml)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/asr24/StoryCraftAI)

Learn how to automate story generation using deep learning and deploy a full-featured web application with CI/CD.

## Project Description

**StoryCraftAI** is an automatic story generation system built entirely from scratch using deep learning. This project covers the entire pipeline—from scraping and preprocessing classic short stories to training various neural network architectures and generating creative narratives from a user-provided seed text.

### Key Features
- **Data Scraping & Preprocessing:**
  - Automatically scrapes classic short stories from [classicshorts.com](http://www.classicshorts.com/).
  - Cleans text by removing unwanted characters and tokenizes sentences into words.
- **Multiple Model Architectures:**
  - **GRU & LSTM:** Optimized for sequence prediction.
  - **Bidirectional Models:** (Bidirectional-LSTM and Bidirectional-GRU) Capture richer context.
  - **Hybrid Approaches:** Combine strengths of different models for better performance.
- **Training & Evaluation:**
  - Generates training sequences (50 words as input and 1 as output).
  - Tracks performance with accuracy and loss metrics.
  - Saves training history for visualization.
- **Interactive Web Application:**
  - Flask-based web interface for real-time story generation.
- **CI/CD & Deployment:**
  - Deployed on Hugging Face Spaces with continuous integration via GitHub Actions.

## Getting Started

### Prerequisites
- Python 3.8 or later.
- Virtual environment (recommended).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/StoryCraftAI.git
   cd StoryCraftAI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK data:
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

5. Run the application locally:
   ```bash
   python app.py
   ```

## Training the Models

### Pipeline Overview
- **Data Scraping:** Fetches stories from [classicshorts.com](http://www.classicshorts.com/).
- **Preprocessing:** Tokenizes text and creates sequences of 51 tokens (50 input, 1 target).
- **Model Building:** Supports architectures like `gru`, `lstm`, `bi_di_gru`, and `bi_di_lstm`.
- **Training:** Uses categorical crossentropy loss. Training logs and performance graphs are generated.

### Execution
Run the training scripts using Jupyter notebooks in the `Notebooks/` folder or directly from the command line. Save the trained model, tokenizer, and training history in the `Models/` directory.

## Deployment

### Using Docker
1. Build and run the Docker container locally:
   ```bash
   docker build -t storycraftai .
   docker run -p 7860:7860 storycraftai
   ```

### CI/CD on Hugging Face Spaces
- Configured for CI/CD using GitHub Actions. Every push to the `main` branch triggers an automated rebuild and redeployment.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a new branch.
3. Submit a pull request detailing your modifications.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or feedback:
- Email: imamitsingh958@gmail.com
- GitHub: [@amitsingh-24](https://github.com/amitsingh-24)
- Linkedin: [@Linkedin](https://www.linkedin.com/in/amit-singh-rajawat-4787a4213/)

Enjoy creating and exploring creative stories with **StoryCraftAI**!
