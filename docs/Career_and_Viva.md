# Career Materials: Resume, LinkedIn, and Viva Prep

## 1. Resume Bullet Points
**AI & NLP Developer | Next Word Prediction System**
- Architected an end-to-end NLP Next Word Prediction engine using TensorFlow/Keras to empirically demonstrate the Vanishing Gradient Problem in sequential data.
- Built and trained parallel sequence models (Simple RNN and LSTM) on text corpora, achieving significantly higher validation accuracy and training stability with the LSTM architecture.
- Developed an interactive Streamlit web application to serve real-time model inference, allowing users to visually compare generation quality and architecture metrics.
- Deployed the complete MLOps pipeline, encompassing data preprocessing (tokenization, n-gram sequencing, padding), model checkpointing, and cloud hosting.

## 2. LinkedIn Post
🚀 **Just finished building an end-to-end Deep Learning project tackling one of AI's most famous bottlenecks: The Vanishing Gradient Problem!**

I built a Context-Aware Next Word Prediction System comparing standard RNNs with LSTMs to see the theory in action. 

🔍 **What I learned & implemented:**
- **The Math:** Why repeated multiplication in Backpropagation Through Time (BPTT) causes RNNs to "forget" past context.
- **The Solution:** How LSTM's Cell State and gating mechanisms (Forget, Input, Output) preserve gradients over long sequences.
- **The Code:** Built custom TensorFlow/Keras models, preprocessing pipelines (Tokenization, Padding), and an interactive UI using Streamlit!

Check out the full code and try the web app on my GitHub: [Link]

#DeepLearning #MachineLearning #NLP #ArtificialIntelligence #TensorFlow #Python #DataScience #Streamlit

---

## 3. Viva & Interview Questions

**Q1: Explain the Vanishing Gradient Problem mathematically.**
**A:** During Backpropagation Through Time (BPTT) in RNNs, the gradient of the loss with respect to weights involves the chain rule, multiplying the weight matrix repeatedly across time steps $t$. If the eigenvalues of the weight matrix are less than 1, multiplying these small numbers together causes the gradient to shrink exponentially toward zero, preventing early layers from updating.

**Q2: How exactly does an LSTM solve this?**
**A:** LSTM introduces a "Cell State" ($C_t$) which runs straight down the entire chain with only minor linear interactions. The update rule for the cell state relies on *addition* rather than purely multiplication ($C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$). This additive property allows gradients to flow backwards unimpeded.

**Q3: What are the three gates in an LSTM?**
**A:** 
1. **Forget Gate:** Uses a sigmoid layer to decide what information to throw away from the cell state.
2. **Input Gate:** A sigmoid layer decides which values to update, and a tanh layer creates a vector of new candidate values.
3. **Output Gate:** Decides what parts of the cell state to output based on a filtered version of the cell state.

**Q4: Why did we use 'Categorical Crossentropy' as the loss function?**
**A:** Because next word prediction is a multi-class classification problem. The vocabulary size represents the number of classes, and we are predicting a probability distribution over these classes.
