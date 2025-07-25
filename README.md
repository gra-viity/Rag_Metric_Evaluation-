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
1. Configure your AWS credentials and region in the `config` dictionary at the top of `metricEval.py`.
2. (Optional) Replace the sample documents, queries, and expected responses with your own data.
3. Run the script:

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

## References
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [LangChain AWS](https://github.com/langchain-ai/langchain-aws)
- [Ragas](https://github.com/explodinggradients/ragas)

---
For questions or issues, please open an issue in this repository.
