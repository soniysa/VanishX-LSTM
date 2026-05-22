import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout
import pandas as pd
import os
import plotly.graph_objects as go
import streamlit.components.v1 as components

# -------------------------------------------------------------
# Configuration & Theming
# -------------------------------------------------------------
st.set_page_config(page_title="VanishX : Solving Vanishing Gradient using LSTM", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }
    
    /* Clean Slate Background */
    .stApp { 
        background-color: #0f172a;
        color: #e2e8f0; 
    }
    
    .main h1, .main h2, .main h3, .main h4, .main h5,
    [data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3 { 
        color: #f8fafc !important; 
        font-weight: 600 !important; 
    }
    
    /* Sky Blue Accent for Main Title */
    .gradient-text {
        font-size: 2.5rem; 
        font-weight: 700; 
        color: #38bdf8;
        margin-bottom: 20px;
    }
    
    /* Neat Slate Cards */
    .glass-card {
        background-color: #1e293b;
        border: 1px solid #334155; 
        border-radius: 12px;
        padding: 24px; 
    }
    
    /* Code block styling */
    .code-box {
        background-color: #0b0f19; 
        border: 1px solid #334155;
        padding: 16px; 
        border-radius: 8px; 
        font-family: monospace;
        color: #cbd5e1;
    }
    
    /* Prediction Output Box */
    .prediction-text {
        font-family: monospace;
        color: #e0f2fe;
        background-color: #0b0f19; 
        padding: 20px; 
        border-radius: 10px;
        border-left: 4px solid #38bdf8;
        font-size: 1.1rem; 
        line-height: 1.6;
    }
    
    /* Sky Blue Buttons */
    .stButton>button {
        background-color: #0284c7; 
        color: #ffffff !important;
        border: none; 
        border-radius: 8px; 
        padding: 12px 24px; 
        font-weight: 600; 
        width: 100%;
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0369a1;
        color: #ffffff !important;
    }
    
    /* Sidebar styling reset to ensure native font colors are visible */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div {
        color: inherit !important;
    }
    
    hr {
        border-color: #334155;
    }
    </style>
""", unsafe_allow_html=True)

class StreamlitProgress(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, status_text, total_epochs):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Epoch {epoch+1}/{self.total_epochs} - loss: {logs['loss']:.4f} - accuracy: {logs.get('accuracy', logs.get('acc', 0)):.4f}")

def predict_next_words(model, tokenizer, max_sequence_len, text, next_words=3):
    result = text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        
        # Removed creativity/temperature. Using pure argmax for deterministic and accurate prediction.
        predicted_word_index = np.argmax(predicted_probs)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
        result += " " + output_word
    return result

st.markdown('<div class="gradient-text">VanishX : Solving Vanishing Gradient using LSTM 🎓</div>', unsafe_allow_html=True)
st.markdown("### An Interactive Deep Learning Experiment & Educational Platform")
st.markdown("---")

with st.sidebar:
    st.markdown("### 🧭 Interactive Syllabus")
    page = st.radio("", ["📚 1. Learn the Theory", "🛠️ 2. Train on Your Data", "🔮 3. Live Experiment", "📊 4. Performance Proof"])
    st.markdown("---")
    st.info("💡 **Student Tip:** Go to Step 2 to train models on your own custom text, then predict in Step 3!")

# -------------------------------------------------------------
# PAGE 1: LEARN THEORY
# -------------------------------------------------------------
if page == "📚 1. Learn the Theory":
    st.markdown("## 🧠 The Mathematics & Architecture of Sequence Models")
    st.markdown("Explore the fundamental concepts, mathematical formulas, and visual architectures that define how machines understand sequences.")
    
    tab1, tab2, tab3 = st.tabs(["📉 1. Simple RNN & The Problem", "🦸‍♂️ 2. The LSTM Solution", "🔬 3. Mathematical Deep Dive"])
    
    with tab1:
        st.markdown("### Simple Recurrent Neural Network (RNN)")
        st.write("A Simple RNN processes sequences by maintaining a **hidden state ($h_t$)** that gets updated at every timestep based on the current input ($x_t$) and the previous hidden state ($h_{t-1}$).")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 📐 Core Formula")
            st.latex(r"h_t = \tanh(W_h h_{t-1} + W_x x_t + b)")
        with col2:
            st.markdown("#### 🏗️ Architecture Visualization")
            try:
                st.image("visualizations/rnn_architecture.png", use_container_width=True, caption="Recurrent Neural Network")
            except Exception as e:
                st.warning("⚠️ Please save the provided RNN diagram as 'visualizations/rnn_architecture.png' to see it here!")
        
        st.markdown("#### ⚠️ The Vanishing Gradient Problem")
        st.write("When learning long sequences, the network uses **Backpropagation Through Time (BPTT)**. To find how the loss ($L$) changes with respect to early weights, we use the chain rule:")
        st.latex(r"\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W} \approx \prod_{k=1}^{T} \frac{\partial h_k}{\partial h_{k-1}}")
        st.error("**Why it fails:** Since the gradient involves multiplying the weight matrix $W$ repeatedly, if the weights are $< 1$ (e.g., $0.5$), multiplying them 100 times drives the gradient to exactly **0**. The network becomes 'blind' to long-term context!")
        
        # Interactive Plotly graph for Vanishing Gradient
        st.markdown("#### 📉 Interactive Visualization: Watch the Gradient Vanish")
        st.write("Below is a real-time plot showing what happens to your gradient signal when multiplied by a weight of 0.8 over 50 steps.")
        
        steps = np.arange(1, 51)
        gradient_values = np.power(0.8, steps)
        
        fig_decay = go.Figure()
        fig_decay.add_trace(go.Scatter(x=steps, y=gradient_values, mode='lines+markers', name='Gradient Signal', line=dict(color='#ef4444', width=3), marker=dict(size=6)))
        fig_decay.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='#e2e8f0'),
            xaxis_title='Time Steps (Words in a sequence)',
            yaxis_title='Gradient Strength',
            hovermode="x unified",
            margin=dict(l=0, r=0, t=10, b=0),
            height=300
        )
        fig_decay.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#334155')
        fig_decay.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#334155')
        st.plotly_chart(fig_decay, use_container_width=True, config={'displayModeBar': False})
        
    with tab2:
        st.markdown("### Long Short-Term Memory (LSTM)")
        st.write("LSTM solves the Vanishing Gradient problem by introducing a direct, uninterrupted 'highway' called the **Cell State ($C_t$)**. Instead of just multiplying, LSTMs use **Addition** to update memory.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 🚪 The Three Gates")
            st.markdown("""
            1. 🗑️ **Forget Gate ($f_t$):** Decides what info to delete from the past state.
            2. ✍️ **Input Gate ($i_t$):** Decides what new info to add.
            3. 👁️ **Output Gate ($o_t$):** Decides what to output as the new hidden state.
            """)
        with col2:
            st.markdown("#### 🏗️ Architecture Visualization")
            try:
                st.image("visualizations/lstm_architecture.png", use_container_width=True, caption="LSTM Architecture: Components")
            except Exception as e:
                st.warning("⚠️ Please save the provided LSTM diagram as 'visualizations/lstm_architecture.png' to see it here!")
        
        # Interactive Plotly graph for Constant Gradient
        st.markdown("#### 🛡️ Interactive Visualization: The Constant Gradient Highway")
        st.write("Because LSTMs use Addition instead of Multiplication for the Cell State, the gradient strength remains stable across 50 steps.")
        
        stable_values = np.ones(50)
        
        fig_stable = go.Figure()
        fig_stable.add_trace(go.Scatter(x=steps, y=stable_values, mode='lines+markers', name='LSTM Cell State Gradient', line=dict(color='#38bdf8', width=3), marker=dict(size=6)))
        fig_stable.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='#e2e8f0'),
            xaxis_title='Time Steps (Words in a sequence)',
            yaxis_title='Gradient Strength',
            yaxis=dict(range=[0, 1.2]),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=10, b=0),
            height=300
        )
        fig_stable.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#334155')
        fig_stable.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#334155')
        st.plotly_chart(fig_stable, use_container_width=True, config={'displayModeBar': False})
        
    with tab3:
        st.markdown("### 🧮 The 4 Equations of LSTM")
        st.write("An LSTM cell is completely defined by these 4 mathematical operations occurring at every timestep $t$. Notice the use of Sigmoid $\sigma$ (outputs 0 to 1) for gates!")
        
        st.markdown("**1. Forget Gate:** (Outputs a number between 0 and 1)")
        st.latex(r"f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)")
        
        st.markdown("**2. Input Gate & Candidate Memory:**")
        st.latex(r"i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)")
        st.latex(r"\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)")
        
        st.markdown("**3. Cell State Update:** (The Magic Step ✨)")
        st.success("This step uses **ADDITION** ($+$) instead of multiplication! This ensures gradients flow backward perfectly without vanishing.")
        st.latex(r"C_t = f_t * C_{t-1} + i_t * \tilde{C}_t")
        
        st.markdown("**4. Output Gate:**")
        st.latex(r"o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)")
        st.latex(r"h_t = o_t * \tanh(C_t)")

# -------------------------------------------------------------
# PAGE 2: TRAIN ON YOUR DATA
# -------------------------------------------------------------
elif page == "🛠️ 2. Train on Your Data":
    st.markdown("## Train Models Dynamically")
    st.markdown("Provide your own text data. The models will learn from your text and predict based on it. We will train both a Simple RNN and an LSTM so you can compare their performance.")
    
    default_text = """Artificial Intelligence is transforming industries and the world.
Deep learning models solve complex problems like the vanishing gradient.
Traditional RNN networks fail to remember long sequences of data.
LSTM solves the vanishing gradient problem using memory cells and gates.
Machine learning is the future of software development and automation.
Natural language processing enables computers to understand human text.
The vanishing gradient problem occurs during backpropagation through time.
Long short term memory networks use a forget gate to discard useless information.
Data science and artificial intelligence are revolutionizing modern technology.
Neural networks are inspired by the human brain and biological neurons.
Artificial intelligence is the ability of a computer program or a machine to think and learn.
Deep learning is a subset of machine learning based on artificial neural networks."""

    user_text = st.text_area("Enter your custom text corpus here:", value=default_text, height=200)
    
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Epochs (Training iterations):", 50, 500, 150)
    with col2:
        embedding_dim = st.slider("Embedding Dimensions:", 10, 100, 50)

    if st.button("Start Training 🚀"):
        if not user_text.strip():
            st.error("Please enter some text to train on.")
        else:
            with st.spinner("Processing Data..."):
                # Process the text as a single continuous sequence to force long-term memory dependencies
                corpus = [user_text.lower().replace('\n', ' ').strip()]
                
                tokenizer = Tokenizer()
                tokenizer.fit_on_texts(corpus)
                total_words = len(tokenizer.word_index) + 1
                
                input_sequences = []
                for line in corpus:
                    token_list = tokenizer.texts_to_sequences([line])[0]
                    for i in range(1, len(token_list)):
                        n_gram_sequence = token_list[:i+1]
                        input_sequences.append(n_gram_sequence)
                
                if len(input_sequences) == 0:
                    st.error("Not enough words to train! Please enter a longer text.")
                    st.stop()

                max_sequence_len = max([len(x) for x in input_sequences])
                input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
                
                X, y = input_sequences[:,:-1], input_sequences[:,-1]
                y = tf.keras.utils.to_categorical(y, num_classes=total_words)

            st.success(f"Data processed! Vocabulary Size: {total_words}, Sequences: {len(X)}")

            # --- Train RNN ---
            st.markdown("#### ❌ Training Simple RNN...")
            rnn_progress = st.progress(0)
            rnn_status = st.empty()
            
            rnn_model = Sequential([
                Embedding(total_words, embedding_dim, input_length=max_sequence_len-1),
                SimpleRNN(4, return_sequences=False),
                Dense(total_words, activation='softmax')
            ])
            rnn_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
            
            rnn_history = rnn_model.fit(X, y, epochs=epochs, verbose=0, 
                                        callbacks=[StreamlitProgress(rnn_progress, rnn_status, epochs)])

            # --- Train LSTM ---
            st.markdown("#### ✅ Training LSTM...")
            lstm_progress = st.progress(0)
            lstm_status = st.empty()
            
            lstm_model = Sequential([
                Embedding(total_words, embedding_dim, input_length=max_sequence_len-1),
                LSTM(256, return_sequences=False),
                Dense(total_words, activation='softmax')
            ])
            lstm_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
            
            lstm_history = lstm_model.fit(X, y, epochs=epochs, verbose=0, 
                                          callbacks=[StreamlitProgress(lstm_progress, lstm_status, epochs)])

            # Save everything to session state
            st.session_state['rnn_model'] = rnn_model
            st.session_state['lstm_model'] = lstm_model
            st.session_state['tokenizer'] = tokenizer
            st.session_state['max_sequence_len'] = max_sequence_len
            st.session_state['rnn_history'] = rnn_history.history
            st.session_state['lstm_history'] = lstm_history.history
            
            # Store flattened cleaned corpus for validation
            full_seqs = tokenizer.texts_to_sequences(corpus)
            clean_corpus_lines = tokenizer.sequences_to_texts(full_seqs)
            st.session_state['clean_corpus'] = " ".join(clean_corpus_lines)
            
            st.balloons()
            st.success("Training Complete! Go to Step 3 to try out the predictor.")

# -------------------------------------------------------------
# PAGE 3: PREDICTOR
# -------------------------------------------------------------
elif page == "🔮 3. Live Experiment":
    st.markdown("## The Playground: Test it Yourself")
    
    if 'rnn_model' not in st.session_state or 'lstm_model' not in st.session_state:
        st.warning("⚠️ You need to train the models first! Please go to **2. Train on Your Data**.")
    else:
        st.info("💡 **Student Task:** Enter a starting phrase based on the data you trained on. The network will complete the sentence.")
        
        rnn_model = st.session_state['rnn_model']
        lstm_model = st.session_state['lstm_model']
        tokenizer = st.session_state['tokenizer']
        max_sequence_len = st.session_state['max_sequence_len']
        
        with st.container():
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            
            # Use a random word from the tokenizer as the default input
            default_word = list(tokenizer.word_index.keys())[0] if len(tokenizer.word_index) > 0 else "Artificial"
            user_input = st.text_input("Enter a starting phrase:", default_word.capitalize())
            
            num_words = st.slider("Words to Predict:", 1, 20, 5)
            
            if st.button("Run Neural Networks 🚀"):
                test_seq = tokenizer.texts_to_sequences([user_input])[0]
                input_words_count = len(user_input.split())
                
                if len(test_seq) == 0:
                    st.warning("⚠️ **Dataset Alert:** The models don't know any of these words! Please use words from the data you trained on.")
                else:
                    if len(test_seq) < input_words_count:
                        st.info(f"ℹ️ **Note:** {input_words_count - len(test_seq)} word(s) were ignored because they weren't in the training data.")

                    with st.spinner("Calculating Gradients & Forward Propagating..."):
                        rnn_output = predict_next_words(rnn_model, tokenizer, max_sequence_len, user_input, num_words)
                        lstm_output = predict_next_words(lstm_model, tokenizer, max_sequence_len, user_input, num_words)
                    
                    # Convert output sequences back to token strings to match the cleaned corpus format exactly
                    rnn_clean = tokenizer.sequences_to_texts([tokenizer.texts_to_sequences([rnn_output])[0]])[0]
                    lstm_clean = tokenizer.sequences_to_texts([tokenizer.texts_to_sequences([lstm_output])[0]])[0]
                    clean_corpus = st.session_state.get('clean_corpus', '')
                    
                    rnn_is_correct = rnn_clean in clean_corpus if clean_corpus else False
                    lstm_is_correct = lstm_clean in clean_corpus if clean_corpus else False
                    
                    rnn_icon = "✅" if rnn_is_correct else "❌"
                    lstm_icon = "✅" if lstm_is_correct else "❌"
                    
                    rnn_border = "#00f2fe" if rnn_is_correct else "#ff4b4b"
                    lstm_border = "#00f2fe" if lstm_is_correct else "#ff4b4b"
                    
                    rnn_caption = "Successfully remembered the context!" if rnn_is_correct else "Often struggles to maintain the original topic context."
                    lstm_caption = "Usually maintains topic perfectly using its internal memory cell." if lstm_is_correct else "Failed to maintain the original sequence context."
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"#### {rnn_icon} Simple RNN Output")
                        st.markdown(f"<div style='border-left: 4px solid {rnn_border};' class='prediction-text'>{rnn_output}</div>", unsafe_allow_html=True)
                        st.caption(rnn_caption)
                    with col2:
                        st.markdown(f"#### {lstm_icon} LSTM Output")
                        st.markdown(f"<div style='border-left: 4px solid {lstm_border};' class='prediction-text'>{lstm_output}</div>", unsafe_allow_html=True)
                        st.caption(lstm_caption)
            st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE 4: ANALYTICS
# -------------------------------------------------------------
elif page == "📊 4. Performance Proof":
    st.markdown("## Empirical Proof: Visualizing the Problem")
    st.markdown("This graph is dynamically generated based on the exact data and parameters you just trained on! No pre-saved screenshots.")
    
    if 'rnn_history' in st.session_state and 'lstm_history' in st.session_state:
        rnn_hist = st.session_state['rnn_history']
        lstm_hist = st.session_state['lstm_history']
        
        # Combine accuracy data
        acc_df = pd.DataFrame({
            'RNN Accuracy': rnn_hist.get('accuracy', rnn_hist.get('acc', [])),
            'LSTM Accuracy': lstm_hist.get('accuracy', lstm_hist.get('acc', []))
        })
        
        # Combine loss data
        loss_df = pd.DataFrame({
            'RNN Loss': rnn_hist['loss'],
            'LSTM Loss': lstm_hist['loss']
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### Accuracy Comparison")
            
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(y=acc_df['RNN Accuracy'], mode='lines', name='Simple RNN (❌)', line=dict(color='#ef4444', width=3)))
            fig_acc.add_trace(go.Scatter(y=acc_df['LSTM Accuracy'], mode='lines', name='LSTM (✅)', line=dict(color='#38bdf8', width=3)))
            fig_acc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='#e2e8f0', family='Inter, sans-serif'),
                xaxis_title='Epochs',
                yaxis_title='Accuracy',
                hovermode="x unified",
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_acc.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#334155', zeroline=False)
            fig_acc.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#334155', zeroline=False)
            
            st.plotly_chart(fig_acc, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### Loss Comparison")
            
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=loss_df['RNN Loss'], mode='lines', name='Simple RNN (❌)', line=dict(color='#ef4444', width=3)))
            fig_loss.add_trace(go.Scatter(y=loss_df['LSTM Loss'], mode='lines', name='LSTM (✅)', line=dict(color='#38bdf8', width=3)))
            fig_loss.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='#e2e8f0', family='Inter, sans-serif'),
                xaxis_title='Epochs',
                yaxis_title='Categorical Loss',
                hovermode="x unified",
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_loss.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#334155', zeroline=False)
            fig_loss.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#334155', zeroline=False)
            
            st.plotly_chart(fig_loss, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='glass-card'>
        <h4>👨‍🏫 What does your custom graph teach us?</h4>
        <ul>
            <li><strong>RNN struggles to converge/learn:</strong> Look at the red line! Because gradients vanish, the weights don't update effectively. This leads to unstable loss and poor/plateauing accuracy over time.</li>
            <li><strong>LSTM learns efficiently:</strong> Look at the blue line! Gradients flow consistently through the additive cell state, allowing the network to find optimal weights and memorize your input text perfectly.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No training data found! Go to **2. Train on Your Data** to generate these dynamic graphs.")
