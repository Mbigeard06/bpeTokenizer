# **bpeTokenizer**
This repository was built following the concepts and structure of [minbpe](https://github.com/karpathy/minbpe) by **Andrej Karpathy**. A huge thank you to **Mr. Karpathy** for his incredible **"Neural Networks: Zero to Hero"** tutorial series, which provided invaluable insights into deep learning, tokenization techniques, and model training.

## **🚀 Overview**
This repository contains two tokenizers implementing **Byte Pair Encoding (BPE)** from scratch:  
- **BasicTokenizer** → A standard BPE tokenizer at the **byte level**.  
- **RegexTokenizer** → A more **advanced tokenizer** that uses **regular expressions (regex) for initial text segmentation**, making it more efficient for **LLM tokenization**.  

---

## **🛠 1 - BasicTokenizer**
A minimal **byte-level tokenizer** that applies **Byte Pair Encoding (BPE)** for **text compression** and **efficient tokenization**.

### **🔹 Features**
✅ **Byte-Level Tokenization** → Converts text into UTF-8 byte sequences.  
✅ **Trainable Vocabulary** → Expands beyond **256 bytes** by merging frequent byte pairs.  
✅ **Efficient Encoding & Decoding** → Supports fast tokenization and text reconstruction.  
✅ **Customizable Vocabulary Size** → Define the number of merges to control token granularity.  

### **📌 Implementation Overview**
- **`train(text, vocab_size)`** → Learns frequent byte merges, updating `vocab` and `merges`.  
- **`encode(text)`** → Tokenizes input text using the learned merges.  
- **`decode(ids)`** → Reconstructs text from tokenized IDs.  

---

## **🧩 2 - RegexTokenizer (LLM Optimized)**
The **RegexTokenizer** is an improved tokenizer that **leverages regular expressions** for **pre-segmentation** before applying BPE. This approach makes it **better suited for training LLMs** like GPT-style architectures.

### **💡 Why Use Regex for LLM Tokenization?**
LLMs process text as tokens, but **basic BPE tokenization does not handle complex structures efficiently**. Using regex **improves tokenization by:**
- **Preserving important structures** (e.g., URLs, hashtags, numbers).  
- **Handling punctuation and special symbols correctly** before merging.  
- **Better multilingual support**, especially for languages without whitespace separation (e.g., Chinese, Japanese).  
- **Reducing unnecessary token splits**, leading to **better compression** and a more **efficient vocabulary**.

### **🔹 Features**
✅ **Regex-based Initial Tokenization** → Uses a **precompiled regex pattern (`GPT4_SPLIT_PATTERN`)** to split text into meaningful segments.  
✅ **Handles Special Tokens** → (e.g., `<|endoftext|>`, mentions `@user`, emojis 😉) for proper model training.  
✅ **Efficient Byte Pair Encoding (BPE) Merging** → Applies **frequency-based token merging** to create an optimal vocabulary.  
✅ **Chunk-wise Tokenization for Compression** → Merges the most frequent subwords while preserving structure.  

### **📌 Implementation Overview**
- **`train(text, vocab_size)`**  
  - Uses **regex** to split text into **chunks**.  
  - Converts text into **UTF-8 byte sequences**.  
  - Merges **frequent byte pairs iteratively** until reaching `vocab_size`.  
  - Updates `merges` dictionary and `vocab`.  

- **`encode(text)`**  
  - Uses **regex for initial tokenization**.  
  - Converts text into **byte sequences**.  
  - Calls `_encode_chunk()` to apply BPE merges.  

- **`decode(ids)`**  
  - Converts **token IDs back into bytes**.  
  - Decodes into **UTF-8 text**, ensuring **accurate reconstruction**.  

---
🔗 References & Credits
	•	Inspired by minbpe by Andrej Karpathy.
	•	Based on concepts from “Neural Networks: Zero to Hero”.
	•	Regex-based tokenization inspired by GPT-4’s approach for structured tokenization.
