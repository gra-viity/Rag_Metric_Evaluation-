# RAG Evaluation Pipeline with AWS Bedrock and Ragas

This project provides a pipeline for evaluating Retrieval-Augmented Generation (RAG) systems using AWS Bedrock models and the Ragas evaluation framework. The script `metricEval.py` demonstrates how to retrieve relevant documents, generate answers using a language model, and evaluate the results with key RAG metrics.

## Features
- Uses AWS Bedrock for both LLM and embedding models via `langchain-aws`.
- Retrieves the most relevant document for each query using cosine similarity.
- Generates answers strictly based on retrieved context.
- Evaluates responses using Ragas metrics: Context Recall, Faithfulness, and Factual Correctness.
- Prints detailed results for each query and the overall evaluation.

## Requirements
- Python 3.8+
- AWS credentials with access to Bedrock models
- Packages: `numpy`, `langchain-aws`, `ragas`, `ragas[llm]`, `ragas[embeddings]`

## Usage
1. Create and activate a Python virtual environment (recommended):

   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure your AWS credentials and region in the `config` dictionary at the top of `metricEval.py`.
4. (Optional) Replace the sample documents, queries, and expected responses with your own data.
5. Run the script:

   ```bash
   python metricEval.py
   ```


## How It Works
- **RAG Class**: Handles document embedding, retrieval, and answer generation using Bedrock LLMs and embeddings.
- **Dataset Generation**: For each query, retrieves the most relevant document and generates an answer. Stores the query, context, response, reference answer, and model ID.
- **Evaluation**: Uses Ragas to compute Context Recall, Faithfulness, and Factual Correctness for the generated answers.

## Customization
- Update the `config` dictionary to use different Bedrock models or regions.
- Replace the sample data with your own domain-specific documents and queries.

## Output
- Prints each query, the generated answer, and the LLM used.
- Prints the Ragas evaluation results at the end.

### Understanding the Metrics
When you run the script, you will see output like:

```
{'context_recall': 0.9286, 'faithfulness': 1.0000, 'factual_correctness(mode=f1)': 0.6993}
```

These metrics are defined as follows:

- **context_recall**: Measures how much of the relevant information from the retrieved context was used in the generated answer. A value close to 1 means the answer covers most of the important context.
- **faithfulness**: Indicates whether the answer is strictly based on the retrieved context, without introducing unsupported information. A value of 1.0 means the answer is fully faithful to the context.
- **factual_correctness (mode=f1)**: Evaluates the factual accuracy of the answer compared to the reference answer, using the F1 score (harmonic mean of precision and recall). A higher value means the answer is more factually correct.

These metrics help you assess the quality of your RAG system in terms of using context, staying faithful to the source, and providing factually correct answers.


## References
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [LangChain AWS](https://github.com/langchain-ai/langchain-aws)
- [Ragas](https://github.com/explodinggradients/ragas)

---
For questions or issues, please open an issue in this repository.
