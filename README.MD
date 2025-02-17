
# 🔥 LLM Overflow: Scalability Testing for Language Model APIs

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-API%20Ready-orange)](https://platform.openai.com)

A high-performance load testing tool designed to stress-test LLM (Large Language Model) APIs like OpenAI, measuring **tokens/second**, **concurrency limits**, and **system stability** under real-world conditions.

---

## 🚀 Features

- ⚡ **Concurrent User Simulation**: Test with 1 to 1000+ virtual users.
- 📊 **Token-Level Metrics**: Track time-to-first-token, end-to-end tokens/sec, and error rates.
- 🛠️ **CLI Powered**: Configure tests via command line—no code changes needed.
- 📈 **Result Export**: Save metrics to JSON for analysis in tools like Grafana or Excel.
- 🤖 **Smart Prompt Generation**: Auto-generate diverse prompts using a built-in corpus.

---

## 🎯 Quick Start

**Prerequisites**:
- Python 3.7+
- OpenAI API key (or equivalent for your LLM provider)


### 1. Run a Basic Test
```bash
python3 main.py      --model codellama:7b \
             --base-url https://api.openai.com/v1 \
             --api-key YOUR_API_KEY \
             --max-users 10
```

### 2. Customize the Load
```bash
python3 main.py 
             --model codellama:7b \
             --base-url https://your-llm-api.com \
             --api-key YOUR_KEY \
             --max-users 50 \
             --step 5 \
             --save-results results.json
```

### 3. Analyze Results
```json
// Sample results.json
[
    {
        "num_users": 10,
        "total_tokens": 6741,
        "total_time": 190.91803090900066,
        "successful_requests": 10,
        "avg_tokens_per_second": 35.30834656058775
    }
]
```

---

## 📊 Metrics Tracked

| Metric                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `Successful Requests`   | ratio of requests completed without errors                                      |
| `Tokens/Second`         | End-to-end throughput including network latency                             |
| `Time to First Token`   | Critical for streaming applications                                         |
| `Max Concurrent Users`  | Users before errors occur or latency exceeds thresholds                     |

---

## 🛠️ Advanced Usage

### Custom Corpus
Create a `corpus.txt` file:
```
The quantum theory suggests...
Blockchain technology enables...
Renaissance art characteristics include...
```

Then run:
```bash
python3 main.py \ 
             --custom-corpus-path corpus.txt \
             --model codellama:7b \
             --base-url https://your-llm-api.com \
             --api-key YOUR_KEY \
             --max-users 50 \
             --step 5 \
             --save-results results.json
```
---

## 🌟 Why This Tool?

- **Real-World Simulation**: Mimics actual user behavior with randomized prompts.
- **Bottleneck Identification**: Discover if your system fails at 10 users or scales to 10,000.
- **Cost Prediction**: Estimate API costs under heavy load before deployment.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

MIT License - Free for commercial and personal use. Made with ❤️ by Farbod.

---

**Star ⭐ this repo if it helps you ship reliable LLM applications!**
```

This README:
- Uses emojis and badges for visual appeal
- Provides clear code examples in boxes
- Explains value proposition upfront
- Includes real-world use cases
- Encourages contribution
- Works well on both desktop and mobile GitHub views

Let me know if you'd like to tweak any section!
