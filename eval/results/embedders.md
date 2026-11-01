# Raw embedding benchmark (143 advice queries, v2 KB)

| model | mode | Hit@1 | 95% CI | Hit@3 | Hit@5 | Hit@8* | MRR | EN@1 | Tanglish@1 | Tamil@1 | dim | ms/query | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mpnet-ml | dense | 0.755 | 0.69–0.83 | 0.902 | 0.937 | 0.972 | 0.834 | 0.77 | 0.50 | 0.75 | 768 | 14.4 | 278M, paraphrase-trained |
| bge-m3 | dense | 0.748 | 0.68–0.82 | 0.902 | 0.930 | 0.958 | 0.834 | 0.76 | 0.75 | 0.50 | 1024 | 23.3 | 568M, strong multilingual incl. Tamil |
| mpnet-ml | hybrid-0.3 | 0.741 | 0.67–0.82 | 0.916 | 0.944 | 0.972 | 0.831 | 0.77 | 0.50 | 0.25 | 768 | 15.6 | 278M, paraphrase-trained |
| bge-m3 | hybrid-0.3 | 0.713 | 0.64–0.78 | 0.909 | 0.937 | 0.965 | 0.816 | 0.73 | 0.75 | 0.25 | 1024 | 21.7 | 568M, strong multilingual incl. Tamil |
| bge-m3 | hybrid | 0.657 | 0.58–0.73 | 0.853 | 0.888 | 0.930 | 0.761 | 0.66 | 0.75 | 0.25 | 1024 | 20.9 | 568M, strong multilingual incl. Tamil |
| mpnet-ml | hybrid | 0.643 | 0.57–0.72 | 0.846 | 0.895 | 0.923 | 0.758 | 0.66 | 0.50 | 0.25 | 768 | 15.6 | 278M, paraphrase-trained |
| bm25 | lexical | 0.510 | 0.43–0.59 | 0.685 | 0.727 | 0.776 | 0.620 | 0.52 | 0.62 | 0.00 | - | 0.2 | no model |

*Hit@8 = how often the right verse is among the 8 candidates Gemini sees (what matters for the chatbot).

Overlapping 95% CIs = no real difference. Tanglish/Tamil subsets are small (n≈6 each): treat as indicative.
