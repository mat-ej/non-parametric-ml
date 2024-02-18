import torch
import torch.nn.functional as F
import torch.nn as nn

# Sample logits for 3 samples and 4 classes
logits = torch.tensor([[2.0, 1.5, 0.1, -1.0],
                       [0.5, -0.2, 2.0, 0.1],
                       [-1.5, 2.5, 1.0, 0.0]])

# True class labels
labels = torch.tensor([0, 2, 1])  # Indices of the correct class for each sample

# 1. Manual Calculation of Cross-Entropy Loss
probabilities = F.softmax(logits, dim=1)
log_probabilities = torch.log(probabilities)
true_prob_log = log_probabilities[range(len(labels)), labels]
cross_entropy_loss_manual = -true_prob_log.mean()

# 2. Using nn.CrossEntropyLoss (combines softmax + NLLLoss)
criterion_ce = nn.CrossEntropyLoss()
cross_entropy_loss_ce = criterion_ce(logits, labels)

# 3. Using nn.functional.nll_loss (assuming softmax applied outside)
log_probabilities = F.log_softmax(logits, dim=1)  # Log of softmax probabilities
cross_entropy_loss_nll = F.nll_loss(log_probabilities, labels)

print("Manual Cross-Entropy Loss:", cross_entropy_loss_manual.item())
print("CrossEntropyLoss:", cross_entropy_loss_ce.item())
print("NLL Loss with Log Softmax:", cross_entropy_loss_nll.item())