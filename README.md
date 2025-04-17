# Intelligent Log Message Classification System

This project implements an intelligent log classification system combining rule-based regex patterns, BERT embeddings  with sklearn algorithm, and large language models (LLMs) to automatically categorize IT system logs into actionable categories like workflow errors or deprecation warnings. The Streamlit-based interface enables both batch CSV processing and real-time log analysis, featuring interactive visualizations for insights into classification patterns and a debug mode for technical validation. By leveraging multiple AI techniques in a tiered approach, it balances speed and accuracy, automating log analysis while providing developers and system administrators with clear, categorized insights for faster troubleshooting and system monitoring.

## 🌟 Features

- **Multi-Model Architecture**
  - Regex pattern matching for known formats
  - BERT embedding model all-MiniLM-L6-v2 with sklearn algorithm
  - LLM (Llama-3-70B / deepseek-r1-distill-llama-70b) contextual understanding (via GROQ)
  
- **Streamlit Web Interface**
  - CSV file with progress tracking
  - Real-time single log classification
  - Interactive data visualizations
  
- **Advanced Analytics**
  - Category distribution charts
  - Source system breakdowns
  - Processing time metrics

- **Developer Friendly**
  - Debug mode with pipeline insights
  - Model configuration details
  - Error handling and validation

## 🖥️ Interface Overview

| Feature                | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **CSV Upload**         | Batch process log files with automatic classification                       |
| **Real-Time Analysis** | Instant classification for individual log messages                          |
| **Visual Dashboard**   | Interactive charts showing category distribution and source breakdown       |
| **Debug Mode**         | Technical insights into classification pipeline and performance metrics     |

## 🧠 Classification Pipeline

1. **Regex Filter**  
   Regex pattern matching for common log formats

2. **ML Model**  
   BERT embedding model (all-MiniLM-L6-v2) with sklearn algorithm

4. **LLM Verification**  
   Llama-3-70B / deepseek-r1-distill-llama-70b model via Groq API for complex cases

5. **Final Output**  
   Categorized logs with visualization and export options

## ⚙️ Configuration

### Environment Variables
```env
GROQ_API_KEY=your_api_key_here
```

### Model Selection
- Default LLM: `llama3-70b-8192 | deepseek-r1-distill-llama-70b`
- BERT Embedding Model: `all-MiniLM-L6-v2`
- ML Model: `lr_model.joblib`

## 🛠️ Troubleshooting

| Issue                          | Solution                                        |
|--------------------------------|-------------------------------------------------|
| Missing CSV columns            | Ensure file contains 'source' and 'log_message' |
| GROQ API errors                | Verify .env file configuration                  |
| Dependency conflicts           | Use Python 3.9+ and requirements.txt            |
| Model loading issues           | Check internet connection for BERT download     |

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact Us

- [![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/mrhamxo)
- [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammad-hamza-khattak/)
- [![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:mr.hamxa942@gmail.com)


