Based on the **Project 4 Specification** (Preference Elicitation), the most directly relevant lectures are:

**1. Primary Lecture: Lecture 7 – Designing and Organizing a User Study**

* **Why it is related:** The core objective of Project 4 is to design a rigorous experimental protocol (Task 3) and implement the interface for participants (Task 4).


* **Key relevant topics covered in Lecture 7:**
* The 6 steps of a user study
* Study design choices: Between-subjects vs. Within-subjects vs. Mixed designs
* Mitigating order/learning effects via counterbalancing or Latin square designs
* Ethical and legal considerations (Informed consent, GDPR compliance)
* Evaluation metrics and hypothesis testing



---

**2. Primary Lecture: Lecture 8 – From Interactive Machine Learning to User Modeling**

* **Why it is related:** Project 4 requires building a movie recommendation feature representation (Task 1) and user preference elicitation system.


* **Key relevant topics covered in Lecture 8:**
* Interactive ML and user feedback loops (observation-level interactions)
* Creating user profiles and content-based filtering using feature representations $\phi(\text{item})$
* Latent user preference vectors $w$ and linear utility formulations ($u^T i$)
* GUI requirements for interactive ML systems



---

**3. Secondary Lecture: Lecture 9 – Advanced Problems in User Modeling**

* **Why it is related:** Project 4 requires extending the Bradley-Terry pairwise preference model to multi-item rankings (Task 2) and inferring the latent parameter $w$.


* **Key relevant topics covered in Lecture 9:**
* Formalizing user models $p(a \mid s, \theta)$
* **Luce choice models** for multi-alternative choice probabilities (the natural multi-item extension of Bradley-Terry / Plackett-Luce)
* Inference of user model parameters (Frequentist vs. Bayesian inference)



---

**4. Additional Relevant Material: Lecture 11 – AI Safety (Reinforcement Learning from Human Feedback)**

* **Why it is related:** Contains the formulation and loss function for the **Bradley-Terry preference model** used in Project 4 ($p(y_1 > y_2) = \frac{\exp(r(y_1))}{\exp(r(y_1)) + \exp(r(y_2))}$).
