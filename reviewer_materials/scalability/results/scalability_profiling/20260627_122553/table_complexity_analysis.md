| Component | Main Operation | Complexity | Scalability Implication |
| --- | --- | --- | --- |
| Task matching | query-meta similarity, dynamic fusion, cosine matching, softmax | O(Kd) | Grows linearly with task count K and representation dimension d. |
| Module aggregation | weighted summation over task-specific module tensors | O(K|P|) | Typically dominates matching because |P| is much larger than d. |
| Task-related memory | store task/meta embeddings and frozen task modules | O(Kd + K|P|) | Memory grows approximately linearly with task number under independent task banks. |