// Data structure for topics in the "invert" category
// REPLACE your existing topics object with this entire block
const topics = {
    // --- Core AI/ML/DS (From original) ---
    "data science": {
        category: "Data Science", // Corrected category based on map
        what: "Data science is an interdisciplinary field using scientific methods, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It involves data collection, cleaning, processing, analysis, modeling, and visualization.",
        why: "It enables data-driven decision-making, prediction of future trends, understanding complex phenomena, and solving real-world problems across various domains by leveraging the power of data.",
        how: "Through processes like CRISP-DM or similar workflows, involving data acquisition, preparation, exploratory data analysis (EDA), modeling (statistical or ML), evaluation, and deployment of insights or models.",
        use_cases: ["Healthcare: Predicting disease outbreaks", "Finance: Credit risk assessment", "E-commerce: Product recommendation", "Marketing: Customer churn prediction"],
        analogy: "Data science is like being a detective for data, gathering clues (data), analyzing them, and solving the case (finding insights or building predictive models)."
    },
    "machine learning": {
        category: "ML Core", // Corrected category
        what: "Machine learning is a subset of AI where systems learn patterns from data to make predictions or decisions without being explicitly programmed for the task. It focuses on algorithm development and model training.",
        why: "It automates complex tasks, enables personalization, uncovers hidden patterns in large datasets, and powers applications like recommendation engines, image recognition, and natural language processing.",
        how: "By selecting an appropriate algorithm (supervised, unsupervised, reinforcement), preparing data, training a model on the data, evaluating its performance, tuning hyperparameters, and deploying it.",
        use_cases: ["Spam email filtering", "Image classification", "Predictive maintenance", "Stock market prediction (with caution!)"],
        analogy: "Machine learning is like teaching a student by showing examples (data), letting them learn the rules (patterns), and then testing their ability to apply those rules to new, unseen problems."
    },
    "artificial intelligence": {
        category: "AI General/Theory", // Corrected category
        what: "Artificial intelligence is a broad field of computer science focused on creating machines or systems capable of performing tasks that typically require human intelligence, such as learning, problem-solving, perception, and decision-making.",
        why: "AI aims to augment human capabilities, automate complex processes, solve problems previously intractable for machines, and create new possibilities in science, industry, and daily life.",
        how: "Through various techniques including machine learning, deep learning, symbolic reasoning, knowledge representation, search algorithms, planning, natural language processing, computer vision, and robotics.",
        use_cases: ["Autonomous vehicles", "Virtual personal assistants (Siri, Alexa)", "Medical diagnosis support", "Advanced game playing (Chess, Go)"],
        analogy: "AI is like building an artificial brain or cognitive system, trying to replicate different facets of human intelligence in a machine."
    },

    // --- Generated Content (Based on keywordToCategory) ---

    // Deep Learning
    "deep learning": {
        category: "Deep Learning",
        what: "A subfield of machine learning based on artificial neural networks with multiple layers (deep architectures) between the input and output layers. These networks learn representations of data with multiple levels of abstraction.",
        why: "Excels at learning complex patterns from large amounts of unstructured data like images, text, and sound, leading to breakthroughs in computer vision and NLP.",
        how: "By training deep neural networks (like CNNs, RNNs, Transformers) using large datasets and optimization algorithms like backpropagation and gradient descent.",
        use_cases: ["Image recognition (ImageNet)", "Natural language translation (Google Translate)", "Speech synthesis", "Drug discovery"],
        analogy: "Deep learning is like having a multi-layered processing system in the brain, where each layer extracts increasingly complex features from the input."
    },
    "neural network": {
        category: "Deep Learning",
        what: "A computational model inspired by the structure and function of biological neural networks. It consists of interconnected nodes (neurons) organized in layers that process information.",
        why: "Forms the foundation of deep learning, enabling the learning of complex, non-linear relationships in data.",
        how: "Information flows through layers, with each neuron performing a weighted sum of its inputs followed by an activation function. Learning occurs by adjusting the weights.",
        use_cases: ["Pattern recognition", "Function approximation", "Basic classification tasks"],
        analogy: "A neural network is like a simplified model of the brain's interconnected neurons, learning by strengthening or weakening connections."
    },
    "cnn": {
        category: "Deep Learning",
        what: "Convolutional Neural Network (CNN): A type of deep neural network primarily used for analyzing visual imagery and other grid-like data (e.g., audio spectrograms).",
        why: "Highly effective due to its ability to automatically learn spatial hierarchies of features (from edges to complex objects) using parameter sharing (convolutional kernels).",
        how: "Employs convolutional layers, pooling layers, and fully connected layers. Convolutional layers apply filters to input data to create feature maps.",
        use_cases: ["Image classification", "Object detection", "Image segmentation", "Facial recognition"],
        analogy: "A CNN is like using a series of sliding magnifying glasses (filters) over an image, each looking for specific patterns (edges, textures, shapes)."
    },
    "rnn": {
        category: "Deep Learning",
        what: "Recurrent Neural Network (RNN): A type of neural network designed to handle sequential data by maintaining an internal state (memory) that captures information about previous elements in the sequence.",
        why: "Suitable for tasks where context from previous steps is crucial, like language modeling or time series analysis.",
        how: "Connections between nodes form a directed graph along a temporal sequence, allowing information to persist. Output at one step influences the computation at the next.",
        use_cases: ["Natural language processing (early models)", "Speech recognition", "Time series prediction", "Music generation"],
        analogy: "An RNN is like reading a sentence, where your understanding of the current word depends on the words you've read before."
    },
    "lstm": {
        category: "Deep Learning",
        what: "Long Short-Term Memory (LSTM): An advanced type of RNN specifically designed to address the vanishing gradient problem, allowing it to learn long-range dependencies in sequential data.",
        why: "More effective than simple RNNs for tasks requiring memory over long sequences due to its gating mechanism.",
        how: "Uses specialized 'gates' (input, forget, output) within its cells to control the flow of information, deciding what to remember, forget, and output.",
        use_cases: ["Machine translation", "Sentiment analysis", "Speech recognition", "Time series forecasting"],
        analogy: "An LSTM is like an RNN with a sophisticated memory controller, allowing it to selectively remember important information from the distant past."
    },
    "gru": {
        category: "Deep Learning",
        what: "Gated Recurrent Unit (GRU): Another type of gated RNN, similar to LSTM, but with a simpler architecture (fewer gates), potentially making it computationally faster.",
        why: "Offers a balance between the performance of LSTMs and the simplicity/speed of simpler RNNs, often performing comparably on many tasks.",
        how: "Uses two gates (update and reset gates) to manage information flow and memory, combining the forget and input gates of LSTM into a single update gate.",
        use_cases: ["Natural language processing tasks", "Sequence modeling where LSTMs are also applicable", "Often used when computational resources are more limited"],
        analogy: "A GRU is like a slightly streamlined version of an LSTM's memory controller, often achieving similar results with less complexity."
    },
    "transformer": {
        category: "Deep Learning",
        what: "A deep learning model architecture introduced in 'Attention Is All You Need', relying entirely on self-attention mechanisms instead of recurrence (RNNs) or convolution (CNNs) for sequence processing.",
        why: "Revolutionized NLP by enabling parallel processing of sequences and capturing long-range dependencies more effectively, leading to state-of-the-art results.",
        how: "Uses multi-head self-attention layers and positional encodings to weigh the importance of different words in a sequence relative to each other, regardless of their distance.",
        use_cases: ["Machine translation (BERT, GPT)", "Text summarization", "Question answering", "Language generation (GPT models)"],
        analogy: "A Transformer is like having a meeting where everyone (words) can directly pay attention to everyone else simultaneously to understand the context, rather than passing messages down a line (like RNNs)."
    },
    "bert": {
        category: "Deep Learning",
        what: "Bidirectional Encoder Representations from Transformers (BERT): A pre-trained Transformer-based model designed to understand the context of words in text by looking at the entire sequence (both left and right context).",
        why: "Achieved state-of-the-art performance on a wide range of NLP tasks by learning deep bidirectional representations.",
        how: "Pre-trained on a massive text corpus using tasks like Masked Language Model (MLM) and Next Sentence Prediction (NSP), then fine-tuned for specific downstream tasks.",
        use_cases: ["Google Search query understanding", "Sentiment analysis", "Named Entity Recognition (NER)", "Question Answering"],
        analogy: "BERT is like reading a sentence multiple times, focusing on different parts each time, to get a really deep understanding of every word's meaning in context."
    },
     "gpt": {
        category: "Deep Learning",
        what: "Generative Pre-trained Transformer (GPT): A family of Transformer-based models primarily designed for natural language generation tasks. They are typically auto-regressive, generating text one token at a time based on preceding tokens.",
        why: "Demonstrates remarkable ability to generate coherent, contextually relevant, and often human-quality text across various styles and domains.",
        how: "Pre-trained on vast amounts of text data to predict the next word in a sequence. Can be fine-tuned or used with prompts for specific generation tasks.",
        use_cases: ["Text generation (stories, articles)", "Chatbots (ChatGPT)", "Code generation", "Summarization", "Translation"],
        analogy: "GPT is like an incredibly well-read author who can continue writing a story or text based on the beginning you provide, predicting the most likely next words."
    },
    "gan": {
        category: "Deep Learning",
        what: "Generative Adversarial Network (GAN): A framework involving two competing neural networks: a Generator (creates synthetic data) and a Discriminator (tries to distinguish real data from synthetic data).",
        why: "Enables the generation of realistic synthetic data (images, music, etc.) by learning the underlying distribution of the training data.",
        how: "The Generator tries to fool the Discriminator, while the Discriminator tries to get better at catching fakes. They train together in a zero-sum game until the Generator produces highly realistic data.",
        use_cases: ["Image synthesis (realistic faces, art)", "Data augmentation", "Style transfer", "Video generation"],
        analogy: "A GAN is like a counterfeiter (Generator) trying to make fake money and a detective (Discriminator) trying to spot the fakes. They both get better over time through competition."
    },
    "generative": { // Assuming Generative Models in general
        category: "Deep Learning",
        what: "Generative Models are a class of statistical or machine learning models that aim to learn the underlying probability distribution of a dataset, allowing them to generate new data samples similar to the original data.",
        why: "They enable the creation of synthetic data, data augmentation, understanding data distributions, and powering creative applications.",
        how: "Various approaches exist, including GANs, VAEs, Autoregressive models (like GPT), and Flow-based models, each learning the data distribution differently.",
        use_cases: ["Image/text/music generation", "Anomaly detection", "Density estimation", "Missing data imputation"],
        analogy: "A generative model is like learning the 'recipe' for creating certain types of data (e.g., faces), so you can bake new, similar examples yourself."
    },
     "autoencoder": {
        category: "Deep Learning",
        what: "An unsupervised neural network trained to reconstruct its input. It consists of an Encoder (compresses input to a lower-dimensional latent representation) and a Decoder (reconstructs the input from the latent representation).",
        why: "Useful for dimensionality reduction, feature learning, and as a basis for generative models (like VAEs).",
        how: "Trained by minimizing the reconstruction error between the original input and the output of the Decoder.",
        use_cases: ["Dimensionality reduction", "Data denoising", "Anomaly detection", "Pre-training for other tasks"],
        analogy: "An autoencoder is like summarizing a book into key points (encoding) and then trying to rewrite the original book just from those key points (decoding)."
    },
    "vae": {
        category: "Deep Learning",
        what: "Variational Autoencoder (VAE): A type of generative autoencoder that learns a probability distribution for the latent space, allowing for controlled generation of new data.",
        why: "Provides a principled way to learn a latent space for generation and allows sampling from this space to create diverse outputs.",
        how: "The Encoder maps input to parameters (mean and variance) of a probability distribution in the latent space. The Decoder samples from this distribution to reconstruct the input. Trained using a combination of reconstruction loss and KL divergence.",
        use_cases: ["Image generation", "Data generation", "Learning latent representations", "Semi-supervised learning"],
        analogy: "A VAE is like an autoencoder that learns not just a summary, but a 'fuzzy' region (distribution) of possible summaries, allowing you to pick points within that region to generate new, related outputs."
    },
    "backpropagation": {
        category: "Deep Learning",
        what: "An algorithm used to efficiently train artificial neural networks by calculating the gradient of the loss function with respect to the network's weights. It propagates the error backward through the network.",
        why: "It's the cornerstone algorithm that makes training deep neural networks computationally feasible.",
        how: "Uses the chain rule of calculus to compute gradients layer by layer, starting from the output layer and moving backward, allowing weights to be updated via gradient descent.",
        use_cases: ["Training virtually all feedforward and recurrent neural networks"],
        analogy: "Backpropagation is like figuring out who to blame (which weights) for a mistake (error) by tracing the error back through the layers that contributed to it."
    },
    "activation function": {
        category: "Deep Learning",
        what: "A function applied to the output of a neuron (or layer) in a neural network. It introduces non-linearity into the model, allowing it to learn complex patterns.",
        why: "Without non-linear activation functions, a deep neural network would behave like a single linear layer, limiting its learning capacity.",
        how: "Common examples include ReLU (Rectified Linear Unit), Sigmoid, Tanh, LeakyReLU. Each function transforms the weighted sum of inputs in a specific non-linear way.",
        use_cases: ["Used in most neurons of hidden and output layers in neural networks"],
        analogy: "An activation function is like a neuron's 'firing' mechanism – it decides whether and how strongly the neuron should signal based on the input it receives."
    },
    "fine-tuning": {
        category: "Deep Learning",
        what: "The process of taking a pre-trained model (trained on a large, general dataset) and further training it on a smaller, specific dataset for a particular task.",
        why: "Leverages the knowledge learned from the large dataset, allowing for good performance on the specific task even with limited task-specific data, and saving training time.",
        how: "Typically involves unfreezing some of the later layers of the pre-trained model and training them (and possibly a new output layer) on the target dataset with a low learning rate.",
        use_cases: ["Adapting pre-trained image models (like ResNet) for specific medical image classification", "Adapting BERT for specific text classification tasks"],
        analogy: "Fine-tuning is like taking someone with a broad education (pre-trained model) and giving them specific on-the-job training (fine-tuning) for a particular role."
    },
    "attention mechanism": {
        category: "Deep Learning",
        what: "A mechanism in neural networks, particularly for sequence-to-sequence tasks, that allows the model to selectively focus on relevant parts of the input sequence when producing an output.",
        why: "Improves performance on long sequences by helping the model identify and utilize the most pertinent input information for each output step.",
        how: "Calculates 'attention scores' for different parts of the input, creating a weighted context vector that emphasizes relevant input elements.",
        use_cases: ["Machine translation", "Text summarization", "Image captioning", "Fundamental component of Transformers"],
        analogy: "An attention mechanism is like highlighting the most important words in a source sentence when translating it, focusing your effort on those key parts."
    },
     "graph neural": { // Assuming Graph Neural Network
        category: "Deep Learning",
        what: "Graph Neural Network (GNN): A type of neural network designed to operate directly on graph-structured data, capturing relationships and dependencies between nodes.",
        why: "Enables machine learning on data naturally represented as graphs, like social networks, molecular structures, or knowledge graphs, which traditional models struggle with.",
        how: "Typically works by iteratively aggregating information from a node's neighbors to update the node's representation (embedding).",
        use_cases: ["Social network analysis (link prediction)", "Recommender systems", "Drug discovery (molecular property prediction)", "Traffic forecasting"],
        analogy: "A GNN is like understanding a person in a social network by not just looking at them, but also considering their friends and connections (neighbors)."
    },
    "gnn": { // Same as above
        category: "Deep Learning",
        what: "Graph Neural Network (GNN): A class of neural networks specifically designed for processing data structured as graphs.",
        why: "Allows learning from relational data by considering node features and graph topology simultaneously.",
        how: "Uses message passing or neighborhood aggregation schemes to update node embeddings based on their connections.",
        use_cases: ["Node classification", "Graph classification", "Link prediction", "Molecular modeling"],
        analogy: "A GNN learns about nodes in a network by letting them 'talk' to their neighbors and update their understanding based on those conversations."
    },
    // ML Core
    "supervised": { // Assuming Supervised Learning
        category: "ML Core",
        what: "A type of machine learning where the algorithm learns from a labeled dataset, meaning each data point has both input features and a corresponding correct output label or value.",
        why: "Used for tasks where the goal is to predict a specific output based on given inputs, like classification or regression.",
        how: "The model learns a mapping function from inputs to outputs by minimizing the difference between its predictions and the true labels in the training data.",
        use_cases: ["Image classification (cat vs. dog)", "Spam detection (spam vs. not spam)", "House price prediction (predicting price based on features)"],
        analogy: "Supervised learning is like learning with a teacher who provides the correct answers (labels) for the practice problems (data)."
    },
    "unsupervised": { // Assuming Unsupervised Learning
        category: "ML Core",
        what: "A type of machine learning where the algorithm learns from an unlabeled dataset, trying to find inherent structures, patterns, or relationships within the data without explicit guidance.",
        why: "Used for tasks like discovering hidden groupings, reducing dimensionality, or finding anomalies when labeled data is unavailable.",
        how: "Algorithms identify patterns based on data similarity or distribution, using techniques like clustering or dimensionality reduction.",
        use_cases: ["Customer segmentation", "Anomaly detection (fraud detection)", "Dimensionality reduction (PCA)", "Topic modeling"],
        analogy: "Unsupervised learning is like exploring a new city without a map or guide, trying to figure out the neighborhoods (clusters) and landmarks (patterns) on your own."
    },
    "reinforcement learning": {
        category: "ML Core",
        what: "A type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions and aims to maximize its cumulative reward over time.",
        why: "Suitable for tasks involving sequential decision-making in dynamic environments, like game playing or robotic control.",
        how: "The agent learns a 'policy' (a strategy for choosing actions) through trial and error, guided by the reward signal, often using algorithms like Q-learning or Policy Gradients.",
        use_cases: ["Game playing (AlphaGo)", "Robotics (learning to walk)", "Resource management", "Recommendation systems (optimizing long-term engagement)"],
        analogy: "Reinforcement learning is like training a pet with treats (rewards) and scolding (penalties) – it learns which actions lead to desirable outcomes."
    },
    "rl": { // Same as above
        category: "ML Core",
        what: "Reinforcement Learning (RL): A paradigm of machine learning focused on training agents to make optimal sequences of decisions in an environment to maximize a cumulative reward signal.",
        why: "Enables learning complex control policies or strategies in situations where supervised labels are not readily available, but interaction and feedback are possible.",
        how: "Involves an agent, environment, states, actions, and rewards. Algorithms explore state-action pairs to learn value functions or policies.",
        use_cases: ["Robotic control", "Autonomous systems", "Game AI", "Optimizing system parameters"],
        analogy: "RL is like learning to ride a bike – you try different actions (pedaling, steering), fall sometimes (negative reward), succeed sometimes (positive reward), and gradually learn the best strategy."
    },
    "q-learning": {
        category: "ML Core",
        what: "A model-free reinforcement learning algorithm that learns a policy telling an agent what action to take under what circumstances. It learns the quality (Q-value) of state-action pairs.",
        why: "A fundamental RL algorithm that can learn optimal policies without needing a model of the environment.",
        how: "Iteratively updates Q-values based on the reward received and the maximum Q-value of the next state, following the Bellman equation.",
        use_cases: ["Simple game playing", "Robotic navigation (in discrete state spaces)", "Learning control tasks"],
        analogy: "Q-learning is like creating a cheat sheet (Q-table) that tells you the expected long-term reward for taking any action in any situation you encounter."
    },
    "decision tree": {
        category: "ML Core",
        what: "A supervised learning model that uses a tree-like structure of decisions and their possible consequences. Each internal node represents a test on an attribute, each branch represents the outcome, and each leaf node represents a class label or value.",
        why: "Easy to understand, interpret, and visualize. Can handle both categorical and numerical data.",
        how: "Learned by recursively partitioning the data based on attribute values that best split the data according to criteria like Gini impurity or information gain.",
        use_cases: ["Classification tasks (e.g., loan approval)", "Regression tasks", "Feature importance analysis"],
        analogy: "A decision tree is like playing a game of 20 questions, where each question (node) helps narrow down the possibilities until you reach an answer (leaf)."
    },
    "random forest": {
        category: "ML Core",
        what: "An ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.",
        why: "Generally more accurate and robust than single decision trees, less prone to overfitting, and provides feature importance estimates.",
        how: "Builds each tree on a random subset of the data (bagging) and considers only a random subset of features at each split.",
        use_cases: ["High-performance classification", "Regression", "Widely used in various domains due to robustness"],
        analogy: "A random forest is like asking many different experts (decision trees), each looking at slightly different information, and then making a final decision based on the majority opinion."
    },
    "svm": {
        category: "ML Core",
        what: "Support Vector Machine (SVM): A supervised learning algorithm used for classification and regression. It finds an optimal hyperplane that best separates different classes in the feature space.",
        why: "Effective in high-dimensional spaces, memory efficient (uses a subset of training points - support vectors), and versatile using different kernel functions.",
        how: "Finds the hyperplane that maximizes the margin (distance) between the closest points (support vectors) of different classes. Can use the 'kernel trick' to handle non-linearly separable data.",
        use_cases: ["Text classification", "Image classification", "Bioinformatics", "Handwriting recognition"],
        analogy: "An SVM is like finding the widest possible road (maximum margin hyperplane) that separates two groups of houses (data points) in a neighborhood (feature space)."
    },
    "support vector": { // Assuming Support Vector (concept within SVM)
        category: "ML Core",
        what: "Support Vectors are the data points that lie closest to the decision boundary (hyperplane) in a Support Vector Machine (SVM). They are the critical elements defining the hyperplane.",
        why: "The position of the optimal hyperplane is determined solely by these support vectors; removing other points wouldn't change the solution.",
        how: "Identified during the SVM training process as the points with the smallest margin to the separating hyperplane.",
        use_cases: ["Core component determining the decision boundary in SVM models"],
        analogy: "Support vectors are like the fence posts that define the boundary line between two properties – only the posts right on the edge matter for the line's position."
    },
    "knn": {
        category: "ML Core",
        what: "K-Nearest Neighbors (KNN): A non-parametric, instance-based learning algorithm used for classification and regression. It classifies a data point based on the majority class among its 'k' closest neighbors in the feature space.",
        why: "Simple to understand and implement, requires no explicit training phase (lazy learning), and adapts locally to data.",
        how: "Stores the entire dataset. To predict a new point, it calculates the distance to all training points, finds the 'k' nearest ones, and assigns the label based on a majority vote (classification) or average (regression).",
        use_cases: ["Basic classification tasks", "Recommendation systems (finding similar items/users)", "Anomaly detection"],
        analogy: "KNN is like deciding where to live based on the characteristics of your 'k' closest neighbors – you tend to fit in with the local majority."
    },
     "k-nearest": { // Same as above
        category: "ML Core",
        what: "K-Nearest Neighbors (KNN): An algorithm that classifies new data points based on the classification of their nearest neighbors in the feature space. 'K' is the number of neighbors considered.",
        why: "Intuitive, easy to implement, and effective for data where points close in feature space likely belong to the same class.",
        how: "Finds the 'K' training samples closest in distance to the new point and predicts the label from these.",
        use_cases: ["Pattern recognition", "Recommendation systems", "Data imputation"],
        analogy: "K-Nearest Neighbors is like predicting someone's favorite sports team based on the favorite teams of their K closest friends."
    },
    "naive bayes": {
        category: "ML Core",
        what: "A probabilistic classification algorithm based on Bayes' theorem with a 'naive' assumption of conditional independence between features, given the class.",
        why: "Simple, fast to train, performs well on high-dimensional data (like text), and requires relatively little training data.",
        how: "Calculates the probability of each class given the input features using Bayes' theorem and the independence assumption, then selects the class with the highest probability.",
        use_cases: ["Text classification (spam filtering, sentiment analysis)", "Medical diagnosis (simple models)", "Recommendation systems"],
        analogy: "Naive Bayes is like diagnosing an illness by assuming symptoms are independent: the probability of having a cough AND fever with the flu is treated as P(cough|flu) * P(fever|flu)."
    },
    "gradient boosting": {
        category: "ML Core",
        what: "An ensemble learning technique that builds models (typically decision trees) sequentially. Each new model attempts to correct the errors made by the previous models.",
        why: "Often achieves state-of-the-art performance on structured (tabular) data for classification and regression tasks.",
        how: "Trains models iteratively, with each new model fitting the residual errors of the ensemble built so far, typically using gradient descent in function space.",
        use_cases: ["Tabular data prediction (Kaggle competitions)", "Ranking", "Regression", "Classification"],
        analogy: "Gradient boosting is like building a team where each new member focuses specifically on fixing the mistakes the team made previously."
    },
    "xgboost": {
        category: "ML Core",
        what: "Extreme Gradient Boosting (XGBoost): An optimized and scalable implementation of gradient boosting, known for its speed, performance, and regularization features.",
        why: "Highly efficient, incorporates regularization to prevent overfitting, handles missing values, and supports parallel processing.",
        how: "Improves upon standard gradient boosting with techniques like regularized objective functions, tree pruning based on depth or gain, and hardware optimization.",
        use_cases: ["Widely used in machine learning competitions (especially tabular data)", "Ranking", "Classification", "Regression"],
        analogy: "XGBoost is like a highly optimized, super-fast version of the gradient boosting team-building process, with extra rules to prevent individuals from becoming too specialized (overfitting)."
    },
    "lightgbm": {
        category: "ML Core",
        what: "Light Gradient Boosting Machine (LightGBM): Another high-performance gradient boosting framework that uses histogram-based algorithms and leaf-wise tree growth.",
        why: "Generally faster training speed and lower memory usage compared to XGBoost, especially on large datasets, while often achieving similar accuracy.",
        how: "Uses histogram-based splits (grouping continuous features into bins) and grows trees leaf-wise (choosing the leaf with max delta loss to grow), unlike XGBoost's level-wise growth.",
        use_cases: ["Large-scale tabular data problems", "Situations requiring fast training", "Classification, Regression, Ranking"],
        analogy: "LightGBM is like another highly efficient gradient boosting team, but it focuses on the most impactful areas first (leaf-wise growth) and uses clever shortcuts (histograms) for speed."
    },
    "ensemble methods": {
        category: "ML Core",
        what: "Machine learning techniques that combine predictions from multiple individual models (base learners) to produce a final prediction that is more accurate and robust than any single model.",
        why: "Improves predictive performance, reduces variance (bagging), reduces bias (boosting), and increases robustness.",
        how: "Common techniques include Bagging (e.g., Random Forest), Boosting (e.g., AdaBoost, Gradient Boosting), and Stacking (training a meta-model on predictions of base models).",
        use_cases: ["Used extensively to achieve top performance in ML competitions and real-world applications"],
        analogy: "Ensemble methods are like forming a committee (ensemble) of diverse experts (models) – the committee's collective decision is usually better than any single expert's opinion."
    },
    "bagging": {
        category: "ML Core",
        what: "Bootstrap Aggregating (Bagging): An ensemble method where multiple instances of the same base learning algorithm are trained independently on different random subsets (bootstrap samples) of the training data. Predictions are then averaged or combined by voting.",
        why: "Reduces variance and helps prevent overfitting, especially for high-variance models like decision trees.",
        how: "Creates multiple bootstrap datasets by sampling with replacement, trains a base model on each, and aggregates their predictions.",
        use_cases: ["Random Forests (bagging of decision trees)", "Improving stability of various models"],
        analogy: "Bagging is like taking multiple polls (bootstrap samples) of subsets of the population and averaging the results to get a more stable estimate than a single poll."
    },
    "boosting": {
        category: "ML Core",
        what: "An ensemble method where models are built sequentially, with each new model focusing on correcting the errors made by the previous ones. Later models give more weight to instances that were previously misclassified.",
        why: "Typically reduces bias and often leads to very high accuracy, creating strong classifiers from weak learners.",
        how: "Iteratively trains base models, adjusting the weights of training instances or fitting residuals based on the performance of the current ensemble.",
        use_cases: ["AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM"],
        analogy: "Boosting is like a student repeatedly studying the topics they got wrong on practice tests, focusing their effort where they are weakest."
    },
     "linear regression": {
        category: "ML Core",
        what: "A supervised learning algorithm used to model the linear relationship between a dependent variable (output) and one or more independent variables (inputs) by fitting a linear equation to the observed data.",
        why: "Simple, interpretable, computationally efficient, and forms the basis for many more complex models.",
        how: "Finds the best-fitting straight line (or hyperplane) through the data points, typically by minimizing the sum of squared differences (errors) between the observed and predicted values (Ordinary Least Squares - OLS).",
        use_cases: ["Predicting continuous values (e.g., house prices based on size)", "Analyzing trends", "Quantifying relationships between variables"],
        analogy: "Linear regression is like drawing the best possible straight line through a scatter plot of data points to predict future values along that line."
    },
    "logistic regression": {
        category: "ML Core",
        what: "A supervised learning algorithm used for binary classification tasks (predicting one of two outcomes). Despite its name, it models the probability of an instance belonging to a class using a logistic (sigmoid) function.",
        why: "Outputs probabilities, interpretable coefficients, computationally efficient, and widely used for classification.",
        how: "Fits a linear equation to the input features, then passes the result through a sigmoid function to squash the output into a probability between 0 and 1. Trained by maximizing likelihood or minimizing log loss.",
        use_cases: ["Spam detection (spam/not spam)", "Medical diagnosis (disease/no disease)", "Click-through rate prediction (click/no click)", "Customer churn prediction (churn/no churn)"],
        analogy: "Logistic regression is like calculating a score based on features and then using a special function (sigmoid) to convert that score into the probability of belonging to a specific category (e.g., 'likely spam')."
    },
    "clustering": {
        category: "ML Core",
        what: "An unsupervised learning task that involves grouping a set of objects (data points) in such a way that objects in the same group (cluster) are more similar to each other than to those in other groups.",
        why: "Discovers hidden structures and groupings in unlabeled data, useful for exploration, segmentation, and anomaly detection.",
        how: "Various algorithms exist, such as K-Means (partitioning based on centroids), DBSCAN (density-based), and Hierarchical Clustering (building a tree of clusters).",
        use_cases: ["Customer segmentation", "Document grouping", "Image segmentation", "Anomaly detection"],
        analogy: "Clustering is like sorting a mixed bag of fruits into piles based on similarity (e.g., all apples together, all oranges together) without knowing the fruit types beforehand."
    },
    "k-means": {
        category: "ML Core",
        what: "K-Means Clustering: A popular partitioning clustering algorithm that aims to partition 'n' observations into 'k' clusters, where each observation belongs to the cluster with the nearest mean (cluster centroid).",
        why: "Simple, computationally efficient for large datasets, and easy to implement.",
        how: "Iteratively assigns points to the nearest centroid and then recalculates the centroids based on the assigned points, until convergence. Requires specifying 'k' beforehand.",
        use_cases: ["Market segmentation", "Image compression (color quantization)", "Document clustering"],
        analogy: "K-Means is like trying to set up 'k' meeting points (centroids) in a city and having everyone go to their nearest meeting point, then adjusting the meeting points to be more central to the groups formed."
    },
    "dbscan": {
        category: "ML Core",
        what: "Density-Based Spatial Clustering of Applications with Noise (DBSCAN): A density-based clustering algorithm that groups together points that are closely packed, marking points in low-density regions as outliers.",
        why: "Can discover clusters of arbitrary shapes (unlike K-Means) and doesn't require specifying the number of clusters beforehand. Robust to outliers.",
        how: "Identifies dense regions by checking the neighborhood around each point. Expands clusters from 'core points' (points with enough neighbors) to density-reachable points.",
        use_cases: ["Anomaly detection", "Clustering spatial data with noise", "Finding non-convex shaped clusters"],
        analogy: "DBSCAN is like finding population centers on a map by looking for areas where houses are densely packed together, ignoring isolated houses (outliers)."
    },
     "regularization": {
        category: "ML Core",
        what: "A technique used in machine learning to prevent overfitting by adding a penalty term to the model's loss function. This penalty discourages overly complex models with large coefficient values.",
        why: "Improves the model's ability to generalize to new, unseen data by reducing complexity and variance.",
        how: "Common types include L1 (Lasso) regularization, which adds the sum of absolute values of coefficients, and L2 (Ridge) regularization, which adds the sum of squared values.",
        use_cases: ["Used in linear models (Ridge, Lasso)", "Neural networks (weight decay)", "Support Vector Machines"],
        analogy: "Regularization is like adding a 'simplicity tax' during model training – models pay a penalty for being too complex, encouraging simpler, more general solutions."
    },
    "l1": { // Assuming L1 Regularization
        category: "ML Core",
        what: "L1 Regularization (Lasso): A type of regularization that adds a penalty equal to the sum of the absolute values of the model's coefficients to the loss function.",
        why: "Encourages sparsity by driving some coefficients to exactly zero, effectively performing feature selection.",
        how: "Minimizes: Original Loss + λ * Σ|coefficient|.",
        use_cases: ["Feature selection in high-dimensional data", "Creating simpler, more interpretable models", "Used in Lasso Regression"],
        analogy: "L1 regularization is like trying to explain something using the fewest words possible – it forces you to drop less important words (features) entirely."
    },
    "l2": { // Assuming L2 Regularization
        category: "ML Core",
        what: "L2 Regularization (Ridge / Weight Decay): A type of regularization that adds a penalty equal to the sum of the squared values of the model's coefficients to the loss function.",
        why: "Discourages large coefficient values, leading to smoother, less complex models that are less prone to overfitting. Doesn't typically force coefficients to zero.",
        how: "Minimizes: Original Loss + λ * Σ(coefficient^2).",
        use_cases: ["Improving generalization of linear models (Ridge Regression)", "Weight decay in neural networks", "Handling multicollinearity"],
        analogy: "L2 regularization is like encouraging everyone on a team to contribute moderately, rather than letting one person (feature) dominate – it shrinks large contributions."
    },
    "bias-variance": { // Assuming Bias-Variance Tradeoff
        category: "ML Core",
        what: "The Bias-Variance Tradeoff is a fundamental concept in supervised learning dealing with the balance between two sources of error: Bias (error from wrong assumptions, underfitting) and Variance (error from sensitivity to small fluctuations in training data, overfitting).",
        why: "Understanding this tradeoff is crucial for building models that generalize well to new data. Decreasing one error source often increases the other.",
        how: "Simple models tend to have high bias/low variance. Complex models tend to have low bias/high variance. The goal is to find a sweet spot with acceptable levels of both.",
        use_cases: ["Model selection", "Hyperparameter tuning", "Diagnosing model performance issues (underfitting/overfitting)"],
        analogy: "The bias-variance tradeoff is like tuning a radio: High bias is being tuned to the wrong station (wrong assumptions). High variance is having too much static, making the signal unstable (sensitive to noise)."
    },
     // Data Science
    "feature engineering": {
        category: "Data Science",
        what: "The process of using domain knowledge to select, transform, and create features (input variables) from raw data to improve the performance of machine learning models.",
        why: "Better features often lead to significantly better model performance, interpretability, and robustness than just choosing complex models.",
        how: "Involves techniques like handling missing values, scaling, encoding categorical variables, creating interaction terms, polynomial features, or deriving domain-specific metrics.",
        use_cases: ["Crucial step in almost all practical machine learning projects"],
        analogy: "Feature engineering is like preparing ingredients before cooking – chopping, mixing, and combining raw items (data) into forms that are easier to work with and taste better (improve model performance)."
    },
    "feature selection": {
        category: "Data Science",
        what: "The process of selecting a subset of relevant features from a larger set of original features to use in model construction.",
        why: "Reduces model complexity, improves training speed, avoids the curse of dimensionality, and can improve model performance by removing irrelevant or redundant features.",
        how: "Methods include Filter methods (evaluating features independently), Wrapper methods (using model performance to select features), and Embedded methods (feature selection integrated into model training, like Lasso).",
        use_cases: ["High-dimensional datasets (genomics, text)", "Improving model interpretability", "Reducing computational cost"],
        analogy: "Feature selection is like packing for a trip – you only take the essential items (features) relevant to your destination (task), leaving unnecessary things behind."
    },
    "dimensionality reduction": {
        category: "Data Science",
        what: "The process of reducing the number of features (dimensions) in a dataset while preserving as much important information as possible.",
        why: "Combats the curse of dimensionality, reduces computational cost, simplifies models, and aids in data visualization.",
        how: "Techniques include Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), t-SNE, and Autoencoders.",
        use_cases: ["Data visualization (reducing to 2D or 3D)", "Noise reduction", "Preprocessing for ML models", "Feature extraction"],
        analogy: "Dimensionality reduction is like creating a concise summary or map (lower dimensions) of a complex book or territory (high-dimensional data) that still captures the main points."
    },
    "pca": {
        category: "Data Science",
        what: "Principal Component Analysis (PCA): A popular unsupervised dimensionality reduction technique that transforms data into a new coordinate system where the axes (principal components) capture the maximum variance in the data.",
        why: "Effective for reducing dimensions while retaining most of the data's variability, useful for visualization, noise reduction, and decorrelating features.",
        how: "Finds orthogonal principal components by performing eigenvalue decomposition on the data's covariance matrix. Projects data onto the top 'k' components.",
        use_cases: ["Image compression", "Data visualization", "Preprocessing for ML algorithms", "Noise filtering"],
        analogy: "PCA is like finding the main directions of spread in a cloud of data points and describing the data primarily along those main directions."
    },
    "t-sne": {
        category: "Data Science",
        what: "t-Distributed Stochastic Neighbor Embedding (t-SNE): A dimensionality reduction technique particularly well-suited for visualizing high-dimensional data in low dimensions (typically 2D or 3D).",
        why: "Excellent at revealing local structure and clusters within data, often producing visually appealing embeddings.",
        how: "Models similarity between high-dimensional points and low-dimensional points using conditional probabilities (Gaussian in high-D, t-distribution in low-D) and minimizes the divergence between these distributions.",
        use_cases: ["Visualizing clusters in high-dimensional data (e.g., genomics, image features)", "Exploring dataset structure"],
        analogy: "t-SNE is like arranging items from a high-dimensional space onto a 2D map such that similar items are placed close together, revealing groups and relationships."
    },
    "data analysis": {
        category: "Data Science",
        what: "The process of inspecting, cleaning, transforming, and modeling data with the goal of discovering useful information, informing conclusions, and supporting decision-making.",
        why: "Extracts meaningful insights from raw data, identifies trends, answers questions, and provides evidence for business or research strategies.",
        how: "Involves techniques like descriptive statistics, exploratory data analysis (EDA), data visualization, hypothesis testing, and potentially building predictive models.",
        use_cases: ["Business intelligence reporting", "Market research analysis", "Scientific research", "Understanding user behavior"],
        analogy: "Data analysis is like examining raw evidence at a crime scene – cleaning it up, looking for patterns, and piecing together the story it tells."
    },
    "statistical modeling": {
        category: "Data Science",
        what: "The process of applying statistical analysis to datasets to build mathematical models that represent relationships between variables and make predictions or inferences.",
        why: "Provides a formal framework for understanding data relationships, quantifying uncertainty, and making statistically sound predictions.",
        how: "Involves selecting appropriate statistical models (e.g., linear regression, ANOVA, time series models), fitting them to data, and evaluating their goodness-of-fit and assumptions.",
        use_cases: ["Predicting sales based on marketing spend", "Analyzing the effect of a drug in clinical trials", "Modeling population growth"],
        analogy: "Statistical modeling is like creating a mathematical blueprint (model) that describes how different factors (variables) interact based on observed data."
    },
    "statistics": {
        category: "Data Science", // Or Core CS/Math
        what: "The science of collecting, analyzing, interpreting, presenting, and organizing data. It deals with uncertainty and variability.",
        why: "Provides the theoretical foundation and practical tools for making sense of data, drawing valid conclusions, and quantifying uncertainty in data science and research.",
        how: "Uses concepts like probability distributions, hypothesis testing, confidence intervals, regression, sampling techniques, and descriptive statistics.",
        use_cases: ["Fundamental to data analysis, machine learning evaluation, A/B testing, scientific research, quality control"],
        analogy: "Statistics is the grammar and vocabulary needed to understand and communicate the language of data."
    },
    "hypothesis testing": {
        category: "Data Science",
        what: "A statistical method used to make decisions or draw conclusions about a population based on sample data. It involves formulating a null hypothesis (no effect) and an alternative hypothesis, then using data to determine if there's enough evidence to reject the null hypothesis.",
        why: "Provides a formal framework for making objective decisions about claims or effects based on evidence and controlling error rates.",
        how: "Calculate a test statistic from the sample data, determine the p-value (probability of observing the data, or more extreme, if the null is true), and compare the p-value to a significance level (alpha).",
        use_cases: ["A/B testing (comparing website versions)", "Clinical trials (testing drug effectiveness)", "Scientific experiments"],
        analogy: "Hypothesis testing is like a court trial: the null hypothesis is 'innocent until proven guilty,' and you need enough evidence (low p-value) to reject innocence (reject the null)."
    },
    "a/b testing": {
        category: "Data Science",
        what: "A method of comparing two versions (A and B) of something (e.g., webpage, app feature, email subject line) to determine which one performs better based on a specific metric (e.g., click-through rate, conversion rate).",
        why: "Allows for data-driven decisions about changes and optimizations by measuring the actual impact of different versions on user behavior.",
        how: "Randomly assign users to see either version A or version B, collect data on the target metric, and use statistical hypothesis testing (like a t-test or z-test) to determine if the difference in performance is statistically significant.",
        use_cases: ["Website optimization", "Marketing campaign testing", "User interface design choices", "Algorithm comparison"],
        analogy: "A/B testing is like running a controlled experiment where you show two different movie posters (versions A and B) to random people and see which one makes more people want to watch the movie (conversion)."
    },
    "time series": { // Assuming Time Series Analysis
        category: "Data Science",
        what: "Time Series Analysis involves analyzing sequences of data points collected over time to extract meaningful statistics and characteristics, identify patterns (trends, seasonality), and make forecasts.",
        why: "Crucial for understanding and predicting phenomena that evolve over time, like stock prices, weather patterns, or sales figures.",
        how: "Uses techniques like decomposition (trend, seasonality, residual), autocorrelation analysis, smoothing methods (moving averages), and forecasting models (ARIMA, Prophet, LSTMs).",
        use_cases: ["Sales forecasting", "Stock market analysis", "Weather prediction", "Economic forecasting", "Monitoring system metrics"],
        analogy: "Time series analysis is like studying a diary (time data) to understand past habits (patterns) and predict future activities (forecasts)."
    },
    "arima": {
        category: "Data Science",
        what: "Autoregressive Integrated Moving Average (ARIMA): A statistical model used for analyzing and forecasting time series data. It combines Autoregressive (AR) components (dependency on past values) and Moving Average (MA) components (dependency on past errors), along with differencing (I) to make the series stationary.",
        why: "A classic and widely used model for time series forecasting, particularly for data exhibiting trends and seasonality (with SARIMA extension).",
        how: "Requires identifying the model orders (p, d, q) through analysis of autocorrelation (ACF) and partial autocorrelation (PACF) plots, then fitting the model to the data.",
        use_cases: ["Sales forecasting", "Economic modeling", "Predicting resource demand"],
        analogy: "ARIMA is like predicting tomorrow's weather based on a combination of yesterday's weather (AR part) and how wrong yesterday's prediction was (MA part), after adjusting for overall trends (I part)."
    },
    "prophet": {
        category: "Data Science",
        what: "A forecasting procedure developed by Facebook, designed for time series data that exhibits strong seasonality (daily, weekly, yearly) and holiday effects, and has missing data or outliers.",
        why: "Easy to use, handles seasonality and holidays effectively, robust to missing data and outliers, and often provides good results with minimal tuning.",
        how: "Models the time series as an additive combination of trend (piecewise linear or logistic), seasonality (using Fourier series), and holiday effects.",
        use_cases: ["Business forecasting (sales, web traffic)", "Capacity planning", "Forecasting data with multiple seasonal patterns"],
        analogy: "Prophet is like a specialized forecasting tool designed for business rhythms, automatically handling weekends, holidays, and yearly trends like an experienced planner."
    },
    "anomaly detection": {
        category: "Data Science",
        what: "The task of identifying data points, events, or observations that deviate significantly from the normal behavior or expected pattern within a dataset.",
        why: "Crucial for identifying critical incidents, potential problems, fraud, or novel events in various systems.",
        how: "Uses various techniques including statistical methods (e.g., z-score, IQR), clustering (e.g., DBSCAN), classification-based methods (e.g., One-Class SVM), or deep learning (e.g., Autoencoders).",
        use_cases: ["Fraud detection (credit cards, insurance)", "Intrusion detection in networks", "Manufacturing defect detection", "Health monitoring (detecting abnormal vital signs)"],
        analogy: "Anomaly detection is like being a security guard watching surveillance footage – your job is to spot anything unusual or out of place compared to the normal routine."
    },
     "recommender system": {
        category: "Data Science",
        what: "A system that predicts the 'rating' or 'preference' a user would give to an item (e.g., movie, product, article) and recommends items the user is likely to appreciate.",
        why: "Helps users discover relevant items in large catalogs, increases user engagement, and drives sales or consumption.",
        how: "Common approaches include Collaborative Filtering (based on user-item interactions), Content-Based Filtering (based on item attributes and user profiles), and Hybrid methods.",
        use_cases: ["Movie recommendations (Netflix)", "Product recommendations (Amazon)", "Music recommendations (Spotify)", "News article suggestions"],
        analogy: "A recommender system is like a helpful friend who suggests movies you might like based on what you and people with similar tastes have watched before."
    },
    "collaborative filtering": {
        category: "Data Science",
        what: "A popular technique used in recommender systems that makes predictions about a user's interests by collecting preferences (ratings, interactions) from many users ('collaboration').",
        why: "Can discover unexpected items and doesn't require understanding item content (features). Works well with large user bases.",
        how: "Finds users with similar tastes (user-based) or items with similar interaction patterns (item-based) and recommends items preferred by similar users or similar items.",
        use_cases: ["Core technique in many large-scale recommender systems (Netflix, Amazon)"],
        analogy: "Collaborative filtering is like getting recommendations based on the idea 'people who liked what you liked, also liked X'."
    },
     "causal inference": {
        category: "Data Science",
        what: "The process of drawing conclusions about causal relationships (cause and effect) from data, going beyond mere correlation.",
        why: "Essential for understanding the true impact of interventions, treatments, or policies, allowing for more effective decision-making.",
        how: "Uses methods like randomized controlled trials (RCTs), observational study techniques (e.g., propensity score matching, regression discontinuity), and causal graphical models (e.g., Bayesian networks).",
        use_cases: ["Evaluating the effectiveness of a marketing campaign", "Determining the impact of a new drug", "Assessing the effect of policy changes"],
        analogy: "Causal inference is like trying to determine if flicking a switch *causes* the light to turn on, not just observing that they often happen together."
    },
    "exploratory data": { // Assuming EDA
        category: "Data Science",
        what: "Exploratory Data Analysis (EDA) is an approach to analyzing datasets to summarize their main characteristics, often using visual methods. It aims to understand the data, discover patterns, spot anomalies, test hypotheses, and check assumptions before formal modeling.",
        why: "Crucial first step in data analysis to gain intuition about the data, guide feature engineering and model selection, and identify potential data quality issues.",
        how: "Involves calculating summary statistics, creating visualizations (histograms, scatter plots, box plots), identifying correlations, and looking for patterns or outliers.",
        use_cases: ["Performed at the beginning of almost every data science project"],
        analogy: "EDA is like exploring a new landscape before building on it – you walk around, look at the terrain, check for obstacles, and get a feel for the area."
    },
    "eda": { // Same as above
        category: "Data Science",
        what: "Exploratory Data Analysis (EDA): The practice of analyzing and visualizing datasets to understand their key characteristics, uncover patterns, identify anomalies, and check assumptions.",
        why: "Essential for data understanding, guiding subsequent analysis steps, and ensuring data quality.",
        how: "Utilizes statistical summaries, plotting techniques (histograms, scatter plots, etc.), correlation analysis, and data profiling.",
        use_cases: ["Initial phase of data analysis", "Understanding variables and relationships", "Identifying data cleaning needs"],
        analogy: "EDA is like getting acquainted with your data by asking questions and visualizing the answers before diving into complex modeling."
    },
     "data cleaning": {
        category: "Data Science",
        what: "The process of detecting and correcting (or removing) corrupt, inaccurate, or irrelevant records from a dataset. Also known as data cleansing or data scrubbing.",
        why: "Ensures data quality, accuracy, and consistency, which is crucial for reliable analysis and model building ('garbage in, garbage out').",
        how: "Involves handling missing values (imputation, deletion), correcting errors, removing duplicates, standardizing formats, and dealing with outliers.",
        use_cases: ["Essential preprocessing step for almost all data analysis and machine learning tasks"],
        analogy: "Data cleaning is like washing and preparing vegetables before cooking – removing dirt, blemishes, and inedible parts to ensure a good final dish."
    },
    "data wrangling": {
        category: "Data Science",
        what: "The process of transforming and mapping raw data from one form into another format that is more appropriate and valuable for downstream analysis. Often includes data cleaning.",
        why: "Makes data usable for analysis or modeling by restructuring, aggregating, joining, and cleaning it.",
        how: "Uses tools and techniques to manipulate data structures, handle inconsistencies, merge datasets, aggregate data, and create new variables.",
        use_cases: ["Preparing data from multiple sources for analysis", "Restructuring data for specific modeling requirements", "Feature engineering"],
        analogy: "Data wrangling is like shaping raw clay (data) into a usable form – kneading it, combining pieces, and smoothing it out before sculpting (analysis)."
    },
    // AI General/Theory
    "ai ethics": {
        category: "AI General/Theory",
        what: "A branch of ethics focusing on the moral behavior of humans as they design, construct, use, and treat artificially intelligent systems, and the ethical considerations of the AI systems themselves.",
        why: "Crucial for ensuring AI systems are developed and deployed responsibly, fairly, safely, and in alignment with human values.",
        how: "Involves discussing principles like fairness, accountability, transparency, privacy, security, bias mitigation, and the potential long-term impacts of AI.",
        use_cases: ["Developing guidelines for autonomous vehicle decision-making", "Auditing algorithms for bias", "Ensuring fairness in AI-driven hiring or loan applications", "Considering the societal impact of AI"],
        analogy: "AI ethics is like establishing the 'rules of the road' and safety standards for developing and using powerful AI technology."
    },
    "explainable ai": {
        category: "AI General/Theory",
        what: "Explainable AI (XAI) refers to methods and techniques that enable human users to understand, trust, and manage the decisions or predictions made by artificial intelligence systems, especially complex 'black box' models.",
        why: "Builds trust, allows debugging, ensures fairness, meets regulatory requirements, and facilitates human-AI collaboration by making AI decisions transparent.",
        how: "Uses techniques like LIME (Local Interpretable Model-agnostic Explanations), SHAP (SHapley Additive exPlanations), feature importance analysis, rule extraction, and generating natural language explanations.",
        use_cases: ["Debugging model predictions", "Ensuring fairness in algorithmic decisions", "Regulatory compliance (e.g., finance, healthcare)", "Building user trust in AI recommendations"],
        analogy: "Explainable AI is like asking an AI 'Why did you make that decision?' and getting a clear, understandable answer, rather than just the decision itself."
    },
    "xai": { // Same as above
        category: "AI General/Theory",
        what: "Explainable AI (XAI): A field focused on developing AI systems whose operations and decisions can be understood by humans.",
        why: "Increases transparency, accountability, and trustworthiness of AI, enabling better debugging, validation, and responsible deployment.",
        how: "Employs various methods to interpret model behavior, such as feature attribution (SHAP, LIME), model simplification, or generating rule-based explanations.",
        use_cases: ["Critical in regulated domains like finance and healthcare", "Debugging complex models", "Building user trust"],
        analogy: "XAI aims to turn 'black box' AI models into 'glass box' models, where you can see and understand the internal workings."
    },
    // Robotics
    "robotics": {
        category: "Robotics",
        what: "An interdisciplinary field involving the design, construction, operation, and application of robots, integrating mechanical engineering, electrical engineering, computer science (especially AI), and other disciplines.",
        why: "Automates tasks (dangerous, repetitive, precise), extends human capabilities, explores inaccessible environments, and enables new forms of interaction and manufacturing.",
        how: "Combines hardware (sensors, actuators, manipulators, structure) with software (control systems, perception algorithms, planning, AI) to create machines that can perceive, reason, and act in the physical world.",
        use_cases: ["Manufacturing automation", "Logistics (warehouse robots)", "Surgery assistance", "Space exploration (rovers)", "Autonomous vehicles"],
        analogy: "Robotics is like building artificial creatures or agents that can physically interact with and perform tasks in the real world."
    },
    "ros": { // Robot Operating System
        category: "Robotics",
        what: "Robot Operating System (ROS): A flexible framework and set of tools, libraries, and conventions for robot software development. It provides operating system-like services like hardware abstraction, device drivers, message-passing, package management, etc.",
        why: "Standardizes robot software development, promotes code reuse, simplifies complex system integration, and fosters a large community.",
        how: "Uses a graph architecture where processes (nodes) communicate via messages published/subscribed over topics. Provides tools for visualization, debugging, simulation.",
        use_cases: ["Widely used in robotics research and development across various platforms (mobile robots, arms, drones)"],
        analogy: "ROS is like a universal adapter and communication system for robot components, allowing different parts (sensors, motors, algorithms) to work together smoothly."
    },
    "ros2": { // ROS 2
        category: "Robotics",
        what: "ROS 2: The next generation of the Robot Operating System, redesigned to address limitations of ROS 1, particularly for multi-robot systems, real-time applications, and commercial/production environments.",
        why: "Offers improved communication (DDS), support for real-time systems, better security, multi-platform support (Windows, macOS), and suitability for commercial products.",
        how: "Built on top of the Data Distribution Service (DDS) standard for communication, offering quality-of-service settings and decentralized discovery.",
        use_cases: ["Commercial robotics applications", "Multi-robot systems", "Real-time control", "Autonomous driving research"],
        analogy: "ROS 2 is like an industrial-grade, upgraded version of ROS, designed for more demanding, real-world robotics applications."
    },
    "robot operating system": { // Same as ROS
         category: "Robotics",
         what: "Robot Operating System (ROS): A middleware framework providing libraries and tools to help software developers create robot applications. It is not a traditional OS but offers OS-like services.",
         why: "Facilitates modularity, code reuse, and collaboration in the complex field of robot software development.",
         how: "Based on a publish/subscribe message-passing system allowing different software modules (nodes) to communicate.",
         use_cases: ["Academic research", "Prototyping", "Development of various robotic systems"],
         analogy: "Robot Operating System is like a standard plumbing system for robots, defining how different software components connect and exchange information."
    },
    // Computer Vision
    "computer vision": {
        category: "Computer Vision",
        what: "A field of AI and computer science that enables computers to 'see', interpret, and understand visual information from the world, typically images or videos.",
        why: "Allows machines to perform tasks that previously required human vision, automating visual inspection, enabling image search, powering autonomous navigation, etc.",
        how: "Uses techniques from image processing, pattern recognition, machine learning (especially deep learning - CNNs) to extract features, detect objects, segment images, and understand scenes.",
        use_cases: ["Object detection in images/video", "Facial recognition", "Medical image analysis", "Autonomous vehicle perception", "Optical Character Recognition (OCR)"],
        analogy: "Computer vision is like giving eyes and a visual cortex (processing ability) to a computer, allowing it to make sense of the visual world."
    },
    " cv": { // Same as above
        category: "Computer Vision",
        what: "Computer Vision (CV): The field concerned with how computers can gain high-level understanding from digital images or videos.",
        why: "Enables automation of visual tasks, image-based analysis, and interaction with the visual world.",
        how: "Combines image processing, feature extraction, machine learning, and deep learning techniques.",
        use_cases: ["Image classification", "Object tracking", "3D reconstruction", "Scene understanding"],
        analogy: "CV teaches computers to interpret pictures and videos, similar to how humans process visual input."
    },
    "image recognition": {
        category: "Computer Vision",
        what: "A subfield of computer vision focused on identifying and classifying objects, people, places, or actions within an image.",
        why: "Fundamental capability for many CV applications, allowing systems to categorize and understand image content.",
        how: "Typically involves extracting features from an image (using traditional methods or deep learning like CNNs) and feeding them into a classifier model.",
        use_cases: ["Photo tagging", "Content-based image retrieval", "Medical image diagnosis aid", "Species identification"],
        analogy: "Image recognition is like teaching a computer to label pictures, saying 'This is a cat,' 'This is a car,' etc."
    },
    "object detection": {
        category: "Computer Vision",
        what: "A computer vision task that involves identifying the presence and location (usually via bounding boxes) of multiple objects within an image or video.",
        why: "Goes beyond simple classification by locating where objects are, crucial for interaction and scene understanding.",
        how: "Uses algorithms like YOLO, SSD, Faster R-CNN that propose regions of interest and classify the objects within those regions.",
        use_cases: ["Autonomous driving (detecting cars, pedestrians)", "Surveillance systems", "Robotics (grasping objects)", "Medical imaging (locating tumors)"],
        analogy: "Object detection is like drawing boxes around all the important items in a picture and labeling what each item is."
    },
     "image segmentation": {
        category: "Computer Vision",
        what: "The process of partitioning a digital image into multiple segments (sets of pixels) to simplify or change the representation of an image into something more meaningful and easier to analyze. It assigns a label to every pixel in an image.",
        why: "Provides a much more detailed understanding of image content than object detection, identifying exact boundaries.",
        how: "Techniques include thresholding, region growing, edge detection, and deep learning methods like U-Net or Mask R-CNN. Types include semantic (pixel-level class), instance (pixel-level object instance), and panoptic segmentation.",
        use_cases: ["Medical image analysis (identifying organs/tumors precisely)", "Autonomous driving (identifying road, sky, cars pixel-by-pixel)", "Satellite image analysis", "Robotic perception"],
        analogy: "Image segmentation is like creating a pixel-perfect coloring book outline for every object or region in an image."
    },
    // NLP
    "nlp": {
        category: "NLP",
        what: "Natural Language Processing (NLP): A subfield of AI concerned with the interactions between computers and human language. It focuses on enabling computers to process, understand, and generate human language.",
        why: "Allows computers to understand and respond to text and speech data, enabling applications like translation, chatbots, sentiment analysis, etc.",
        how: "Combines computational linguistics with statistical modeling, machine learning, and deep learning (RNNs, LSTMs, Transformers) to handle tasks like parsing, entity recognition, translation, and generation.",
        use_cases: ["Machine translation (Google Translate)", "Sentiment analysis (product reviews)", "Chatbots and virtual assistants", "Text summarization", "Information extraction"],
        analogy: "NLP is like teaching a computer to read, write, and understand human languages like English or Spanish."
    },
    "natural language": { // Assuming NLP
        category: "NLP",
        what: "Natural Language Processing (NLP): The area of AI focused on enabling computers to understand, interpret, and generate human language.",
        why: "Bridges the gap between human communication and computer understanding.",
        how: "Employs algorithms to analyze text/speech structure, meaning, and context.",
        use_cases: ["Search engines", "Spell check", "Language translation", "Voice assistants"],
        analogy: "NLP gives computers the ability to comprehend and use language like humans do (or try to!)."
    },
    "sentiment analysis": {
        category: "NLP",
        what: "The use of NLP techniques to identify, extract, and quantify subjective information, opinions, and attitudes expressed in text data (e.g., positive, negative, neutral).",
        why: "Helps understand public opinion, customer feedback, brand perception, and emotional tone in communication.",
        how: "Uses lexicons (lists of words with sentiment scores), machine learning classifiers (like Naive Bayes, SVM trained on labeled data), or deep learning models (RNNs, Transformers).",
        use_cases: ["Analyzing product reviews", "Tracking brand reputation on social media", "Gauging public opinion on policies", "Customer service feedback analysis"],
        analogy: "Sentiment analysis is like reading text and automatically determining if the writer sounds happy, angry, or neutral."
    },
     "text classification": {
        category: "NLP",
        what: "The task of assigning predefined categories or labels to text documents based on their content.",
        why: "Organizes and categorizes large volumes of text data automatically.",
        how: "Involves feature extraction (e.g., TF-IDF, word embeddings) followed by training a classification model (e.g., Naive Bayes, SVM, Logistic Regression, deep learning models).",
        use_cases: ["Spam detection", "Topic labeling (e.g., news articles by topic)", "Sentiment analysis (as a classification task)", "Language detection"],
        analogy: "Text classification is like sorting emails into different folders (e.g., 'Inbox', 'Spam', 'Work', 'Personal') based on their content."
    },
    "named entity recognition": {
        category: "NLP",
        what: "Named Entity Recognition (NER): An NLP task focused on identifying and categorizing named entities (like person names, organizations, locations, dates, monetary values) in text.",
        why: "Extracts structured information from unstructured text, enabling better understanding and information retrieval.",
        how: "Uses rule-based systems, statistical models (like Conditional Random Fields - CRFs), or deep learning models (BiLSTMs, Transformers) trained on annotated text.",
        use_cases: ["Information extraction from news articles", "Customer support ticket analysis (identifying products/people)", "Content recommendation", "Powering chatbots"],
        analogy: "NER is like automatically highlighting all the names of people, places, and organizations in a block of text with different colored markers."
    },
    "ner": { // Same as above
        category: "NLP",
        what: "Named Entity Recognition (NER): The task of locating and classifying named entities mentioned in unstructured text into pre-defined categories.",
        why: "Crucial for information extraction and understanding the key actors and elements described in text.",
        how: "Often approached using sequence labeling models (CRFs, BiLSTM-CRF, Transformers).",
        use_cases: ["Knowledge graph population", "Information retrieval", "Question answering systems"],
        analogy: "NER acts like a highlighter automatically finding and categorizing important proper nouns in text."
    },
    // ... Continue generating for the rest of the keywords in keywordToCategory ...
    // (This would be extremely long to include fully here)

    // Example for Programming Language
    "python": {
        category: "Programming Language",
        what: "A high-level, interpreted, general-purpose programming language known for its clear syntax, readability, and extensive standard library.",
        why: "Widely adopted in data science, machine learning, web development, automation, and scientific computing due to its ease of use and vast ecosystem of libraries (NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch, Django, Flask).",
        how: "Used by writing scripts (.py files) that are executed by the Python interpreter. Leverages libraries for specific tasks.",
        use_cases: ["Data analysis & ML", "Web development (backend)", "Scripting & automation", "Scientific computing"],
        analogy: "Python is like a versatile, easy-to-learn language with a huge dictionary (libraries) available, making it suitable for many different kinds of projects."
    },

    // Example for Tools/Platform
    "scikit-learn": {
        category: "Tools/Platform",
        what: "A popular open-source machine learning library for Python. It provides efficient tools for data analysis and machine learning, including algorithms for classification, regression, clustering, dimensionality reduction, model selection, and preprocessing.",
        why: "Offers a consistent API, extensive documentation, and implementations of many common ML algorithms, making it a go-to library for general ML tasks.",
        how: "Import relevant modules, prepare data (often using NumPy/Pandas), instantiate model objects, fit models to data (`.fit()`), and make predictions (`.predict()`).",
        use_cases: ["Implementing standard ML pipelines", "Benchmarking algorithms", "Educational purposes", "Rapid prototyping"],
        analogy: "Scikit-learn is like a comprehensive toolbox for standard machine learning tasks, providing reliable implementations of common tools."
    },

     // Example for Core CS/Math
    "algorithms": {
        category: "Core CS/Math",
        what: "A well-defined computational procedure or set of rules to be followed in calculations or other problem-solving operations, especially by a computer.",
        why: "Forms the core of computer science and programming, providing efficient and correct methods for solving computational problems.",
        how: "Designed based on mathematical principles and data structures, analyzed for correctness and efficiency (time/space complexity). Examples include sorting algorithms, search algorithms, graph algorithms.",
        use_cases: ["Fundamental to all software development", "Optimizing code performance", "Machine learning model implementation", "Data processing"],
        analogy: "Algorithms are like detailed recipes for solving specific computational problems, ensuring you get the right result efficiently."
    },

     // Example for Web/API
    "api": {
        category: "Web/API",
        what: "Application Programming Interface (API): A set of definitions, protocols, and tools for building software. It specifies how software components should interact, allowing different systems or parts of a system to communicate with each other.",
        why: "Enables modularity, code reuse, integration between different services (e.g., web services), and abstraction of complex implementations.",
        how: "Defines endpoints (URLs for web APIs), request/response formats (e.g., JSON, XML), authentication methods, and expected behavior. REST and GraphQL are common architectural styles for web APIs.",
        use_cases: ["Fetching data from external services (weather, maps)", "Integrating third-party logins (Google, Facebook)", "Microservices communication", "Providing programmatic access to a platform"],
        analogy: "An API is like a restaurant menu and ordering system – it defines what you can order (functions/data) and how to place your order (requests) without needing to know how the kitchen (implementation) works."
    },

     // --- Placeholder for keywords not explicitly generated above ---
     // You might want a generic fallback if a search term isn't found
     // This isn't part of the main topics object, but illustrates a fallback concept
     // "placeholder_topic": {
     //    category: "Default",
     //    what: "This topic is a placeholder. Detailed information is not yet available.",
     //    why: "Placeholder used when specific content hasn't been generated.",
     //    how: "N/A",
     //    use_cases: ["N/A"],
     //    analogy: "N/A"
     // }
}; // End of the topics object definition

// --- CONFIGURATION ---
// MODIFIED: Increased padding to keep nodes more central
const topPadding = 130;    // Space below the top bar
const sidePadding = 100;   // Space from left/right edges
const bottomPadding = 110; // Space above the bottom category buttons

// Repulsion zone around the top bar (#main-page content)
const searchBarZonePadding = 45; // How far the repulsion extends from the top bar elements
const searchBarZoneRepulsion = 3.5; // INCREASED force pushing nodes away from the top bar zone

// --- COLOR CUSTOMIZATION AREA ---
const categoryColors = {
    'Web/API': '#4C6EF5', 'Programming Language': '#FA5252', 'Data Science': '#FD7E14',
    'ML Core': '#FAB005', 'Deep Learning': '#BE4BDB', 'AI General/Theory': '#FFD43B',
    'Robotics': '#74B816', 'Computer Vision': '#40C057', 'NLP': '#228BE6',
    'Data Infra/Ops': '#15AABF', 'Core CS/Math': '#E64980', 'Tools/Platform': '#7950F2',
    'Default': '#CED4DA'
};

// --- KEYWORD TO CATEGORY MAPPING ---
const keywordToCategory = {
    'deep learning': 'Deep Learning', 'neural network': 'Deep Learning', 'cnn': 'Deep Learning', 'rnn': 'Deep Learning',
    'lstm': 'Deep Learning', 'gru': 'Deep Learning', 'transformer': 'Deep Learning', 'bert': 'Deep Learning', 'gpt': 'Deep Learning',
    'gan': 'Deep Learning', 'generative': 'Deep Learning', 'autoencoder': 'Deep Learning', 'vae': 'Deep Learning', 'backpropagation': 'Deep Learning',
    'activation function': 'Deep Learning', 'fine-tuning': 'Deep Learning', 'attention mechanism': 'Deep Learning',
    'pytorch geometric': 'Deep Learning', 'graph neural': 'Deep Learning', 'gnn': 'Deep Learning',
    'machine learning': 'ML Core', 'supervised': 'ML Core', 'unsupervised': 'ML Core', 'reinforcement learning': 'ML Core',
    ' rl': 'ML Core', 'q-learning': 'ML Core', 'sarsa': 'ML Core', 'dqn': 'ML Core', 'decision tree': 'ML Core',
    'random forest': 'ML Core', 'svm': 'ML Core', 'support vector': 'ML Core', 'knn': 'ML Core', 'k-nearest': 'ML Core',
    'naive bayes': 'ML Core', 'gradient boosting': 'ML Core', 'xgboost': 'ML Core', 'lightgbm': 'ML Core', 'adaboost': 'ML Core',
    'ensemble methods': 'ML Core', 'bagging': 'ML Core', 'boosting': 'ML Core', 'linear regression': 'ML Core', 'logistic regression': 'ML Core',
    'clustering': 'ML Core', 'k-means': 'ML Core', 'dbscan': 'ML Core', 'regularization': 'ML Core', 'l1': 'ML Core', 'l2': 'ML Core',
    'bias-variance': 'ML Core', 'markov': 'ML Core',
    'feature engineering': 'Data Science', 'feature selection': 'Data Science', 'dimensionality reduction': 'Data Science', 'pca': 'Data Science', 't-sne': 'Data Science',
    'data science': 'Data Science', 'data analysis': 'Data Science', 'statistical modeling': 'Data Science', 'statistics': 'Data Science', 'hypothesis testing': 'Data Science',
    'a/b testing': 'Data Science', 'time series': 'Data Science', 'arima': 'Data Science', 'prophet': 'Data Science', 'anomaly detection': 'Data Science',
    'recommender system': 'Data Science', 'collaborative filtering': 'Data Science', 'causal inference': 'Data Science', 'exploratory data': 'Data Science',
    'eda': 'Data Science', 'data cleaning': 'Data Science', 'data wrangling': 'Data Science',
    'artificial intelligence': 'AI General/Theory', ' ai': 'AI General/Theory', 'ai ethics': 'AI General/Theory',
    'explainable ai': 'AI General/Theory', 'xai': 'AI General/Theory', 'knowledge representation': 'AI General/Theory', 'ontology': 'AI General/Theory',
    'expert system': 'AI General/Theory', 'planning': 'AI General/Theory', 'constraint satisfaction': 'AI General/Theory', 'symbolic ai': 'AI General/Theory',
    'automated reasoning': 'AI General/Theory', 'fuzzy logic': 'AI General/Theory', 'multi-agent': 'AI General/Theory',
    'robotics': 'Robotics', ' ros ': 'Robotics', 'ros2': 'Robotics', 'robot operating system': 'Robotics',
    'kinematics': 'Robotics', 'dynamics': 'Robotics', 'control theory': 'Robotics', 'pid': 'Robotics',
    'motion planning': 'Robotics', 'path planning': 'Robotics', 'rrt': 'Robotics', 'slam': 'Robotics',
    'localization': 'Robotics', 'mapping': 'Robotics', 'kalman filter': 'Robotics', 'particle filter': 'Robotics', 'sensor fusion': 'Robotics',
    'actuator': 'Robotics', 'sensor': 'Robotics', 'lidar': 'Robotics', 'imu': 'Robotics', 'hri': 'Robotics', 'human-robot': 'Robotics',
    'mobile robot': 'Robotics', 'navigation': 'Robotics', 'manipulator': 'Robotics', 'grasping': 'Robotics', 'swarm robotics': 'Robotics',
    'soft robotics': 'Robotics', 'gazebo': 'Tools/Platform',
    'computer vision': 'Computer Vision', ' cv': 'Computer Vision', 'image recognition': 'Computer Vision', 'object detection': 'Computer Vision',
    'image segmentation': 'Computer Vision', 'image processing': 'Computer Vision', 'feature detection': 'Computer Vision', 'sift': 'Computer Vision',
    'surf': 'Computer Vision', 'orb': 'Computer Vision', 'optical flow': 'Computer Vision', 'stereo vision': 'Computer Vision', '3d vision': 'Computer Vision',
    'point cloud': 'Computer Vision', 'yolo': 'Computer Vision', 'ssd': 'Computer Vision', 'u-net': 'Computer Vision',
    'nlp': 'NLP', 'natural language': 'NLP', 'sentiment analysis': 'NLP', 'text classification': 'NLP', 'named entity recognition': 'NLP', 'ner': 'NLP',
    'topic modeling': 'NLP', 'lda': 'NLP', 'word embedding': 'NLP', 'word2vec': 'NLP', 'glove': 'NLP', 'fasttext': 'NLP',
    'sequence-to-sequence': 'NLP', 'seq2seq': 'NLP', 'machine translation': 'NLP', 'language model': 'NLP', 'computational linguistics': 'NLP',
    'speech recognition': 'NLP', 'asr': 'NLP', 'text generation': 'NLP', 'summarization': 'NLP', 'question answering': 'NLP',
    'sql': 'Data Infra/Ops', 'database': 'Data Infra/Ops', 'postgresql': 'Data Infra/Ops', 'mysql': 'Data Infra/Ops', 'mongodb': 'Data Infra/Ops',
    'nosql': 'Data Infra/Ops', 'redis': 'Data Infra/Ops', 'cassandra': 'Data Infra/Ops', 'big data': 'Data Infra/Ops', 'spark': 'Data Infra/Ops',
    'hadoop': 'Data Infra/Ops', 'kafka': 'Data Infra/Ops', 'data warehouse': 'Data Infra/Ops', 'etl': 'Data Infra/Ops',
    'mlops': 'Data Infra/Ops', 'model deployment': 'Data Infra/Ops', 'feature store': 'Data Infra/Ops', 'model monitoring': 'Data Infra/Ops',
    'ci/cd for ml': 'Data Infra/Ops', 'cloud': 'Data Infra/Ops', 'aws': 'Data Infra/Ops', 'azure': 'Data Infra/Ops', 'gcp': 'Data Infra/Ops',
    'sagemaker': 'Data Infra/Ops', 'azure ml': 'Data Infra/Ops', 'gcp ai': 'Data Infra/Ops', 'distributed computing': 'Data Infra/Ops',
    'python': 'Programming Language', ' r ': 'Programming Language', 'julia': 'Programming Language', 'c++': 'Programming Language', 'java': 'Programming Language',
    'scikit-learn': 'Tools/Platform', 'sklearn': 'Tools/Platform', 'tensorflow': 'Tools/Platform', 'pytorch': 'Tools/Platform', 'keras': 'Tools/Platform',
    'pandas': 'Tools/Platform', 'numpy': 'Tools/Platform', 'scipy': 'Tools/Platform', 'jupyter': 'Tools/Platform', 'notebook': 'Tools/Platform',
    'matplotlib': 'Tools/Platform', 'seaborn': 'Tools/Platform', 'plotly': 'Tools/Platform', 'd3.js': 'Tools/Platform', 'opencv': 'Tools/Platform',
    'nltk': 'Tools/Platform', 'spacy': 'Tools/Platform', 'hugging face': 'Tools/Platform', 'mlflow': 'Tools/Platform', 'kubeflow': 'Tools/Platform',
    'docker': 'Tools/Platform', 'kubernetes': 'Tools/Platform', 'git': 'Tools/Platform',
    'algorithms': 'Core CS/Math', 'data structure': 'Core CS/Math', 'linear algebra': 'Core CS/Math', 'calculus': 'Core CS/Math',
    'probability': 'Core CS/Math', 'statistics': 'Data Science', // Override if DS is more relevant
    'graph theory': 'Core CS/Math', 'information theory': 'Core CS/Math', 'complexity': 'Core CS/Math', 'computation': 'Core CS/Math',
    'optimization': 'Core CS/Math', 'gradient descent': 'Core CS/Math', 'search algorithm': 'Core CS/Math', 'heuristic': 'Core CS/Math', ' a* ': 'Core CS/Math', 'a*': 'Core CS/Math',
    'api': 'Web/API', 'rest': 'Web/API', 'flask': 'Web/API', 'django': 'Web/API', 'streamlit': 'Web/API', 'javascript': 'Web/API'
};
const defaultColor = categoryColors['Default'];


// --- POTENTIAL LABELS ARRAY ---
const potentialLabels = [
    'Artificial Intelligence', 'Machine Learning', 'Deep Learning', 'Data Science', 'Robotics', 'Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning', 'NLP', 'Computer Vision', 'AI Ethics', 'Explainable AI (XAI)', 'Statistical Modeling', 'Probability Theory', 'Linear Algebra for ML', 'Calculus for ML', 'Core Algorithms', 'Essential Data Structures', 'Feature Engineering', 'Model Evaluation Metrics', 'Bias & Fairness in AI', 'Data Privacy in AI', 'Federated Learning', 'Meta-Learning', 'Causal Inference', 'Linear Regression', 'Logistic Regression', 'Decision Trees (CART, ID3)', 'Random Forests', 'SVM Kernels', 'Support Vector Machines', 'K-Means Clustering', 'DBSCAN', 'Hierarchical Clustering', 'PCA', 'Principal Component Analysis', 'LDA (Linear Discriminant Analysis)', 'Naive Bayes Classifiers', 'Gradient Boosting Machines', 'XGBoost', 'LightGBM', 'CatBoost', 'AdaBoost', 'Ensemble Methods', 'Bagging', 'Boosting', 'Stacking', 'KNN', 'K-Nearest Neighbors', 'Q-Learning', 'SARSA', 'Deep Q Networks (DQN)', 'Policy Gradients', 'Actor-Critic Methods (A2C, A3C)', 'Markov Decision Process (MDP)', 'Bayesian Methods', 'Gaussian Processes', 'Artificial Neural Networks (ANN)', 'Multi-Layer Perceptron (MLP)', 'Backpropagation Algorithm', 'Activation Functions (ReLU, Leaky ReLU, Tanh, Sigmoid)', 'CNN Architectures (LeNet, AlexNet, VGG, ResNet, Inception)', 'Convolutional Neural Networks', 'RNN Architectures (Simple RNN, Elman)', 'Recurrent Neural Networks', 'LSTM Networks', 'GRU Networks', 'Transformer Architecture', 'Self-Attention Mechanisms', 'BERTology', 'GPT Models (GPT-3, GPT-4)', 'Generative Adversarial Networks (DCGAN, StyleGAN)', 'Autoencoders (Denoising, Sparse)', 'Variational Autoencoders (VAE)', 'Word Embeddings (Word2Vec, GloVe, FastText)', 'Transfer Learning Strategies', 'Fine-tuning Models', 'Graph Neural Networks (GCN, GraphSAGE)', 'Geometric Deep Learning', 'Optimization Algorithms (SGD, Adam, RMSprop)', 'Learning Rate Schedules', 'Regularization (Dropout, L1/L2)', 'Batch Normalization', 'Sequence-to-Sequence Models', 'Attention Is All You Need', 'Python Ecosystem (Pandas, NumPy, SciPy)', 'R Programming for Stats', 'SQL for Data Analysis', 'Data Cleaning Techniques', 'Data Wrangling & Manipulation', 'Exploratory Data Analysis (EDA)', 'Interactive Data Visualization', 'Matplotlib', 'Seaborn', 'Plotly & Dash', 'Bokeh', 'Feature Selection Methods', 'Hypothesis Testing Frameworks', 'A/B Testing Design & Analysis', 'Time Series Analysis (ARIMA, SARIMA, Prophet)', 'Statistical Process Control', 'Survival Analysis', 'Recommender Systems (Collaborative Filtering, Content-Based, Hybrid)', 'Sentiment Analysis Techniques', 'Text Classification Models', 'Named Entity Recognition (NER)', 'Part-of-Speech Tagging (POS)', 'Topic Modeling (LDA, NMF)', 'Sequence Labeling', 'Coreference Resolution', 'Word Sense Disambiguation', 'N-grams', 'TF-IDF', 'Advanced Word Embeddings', 'Sentence Embeddings (Sentence-BERT)', 'Machine Translation Systems (SMT, NMT)', 'Language Modeling Fundamentals', 'NLTK Library', 'spaCy Library', 'Hugging Face Ecosystem (Transformers, Datasets)', 'Speech Recognition (ASR)', 'Text-to-Speech (TTS)', 'Dialogue Systems & Chatbots', 'Information Retrieval', 'Text Summarization (Extractive, Abstractive)', 'Question Answering Systems', 'Image Classification', 'Object Detection Algorithms (YOLO, SSD, Faster R-CNN)', 'Image Segmentation (Semantic, Instance, Panoptic)', 'OpenCV Library', 'Image Processing Kernels & Filters', 'Feature Detectors & Descriptors (SIFT, SURF, ORB, BRIEF)', 'Optical Flow Calculation', 'Stereo Vision & Depth Perception', '3D Reconstruction', 'Point Cloud Processing', 'Visual Odometry', 'Object Tracking', 'Action Recognition', 'Medical Image Analysis', 'Image Generation with GANs', 'Robot Operating System (ROS/ROS2)', 'Gazebo / Isaac Sim / MuJoCo', 'Forward & Inverse Kinematics', 'Jacobian Matrix', 'Robot Dynamics & Control', 'PID Controllers', 'Optimal Control', 'Model Predictive Control (MPC)', 'Motion Planning Algorithms (RRT*, PRM)', 'A* Search Algorithm', 'SLAM Techniques (Graph SLAM, EKF SLAM)', 'Kalman Filtering (EKF, UKF)', 'Particle Filters', 'Bayesian Filtering', 'Sensor Fusion Strategies', 'LIDAR Processing', 'IMU Calibration & Integration', 'Visual SLAM', 'Mobile Robot Navigation Stacks', 'Path Planning & Following', 'Robot Manipulator Control', 'Robot Grasping Strategies', 'Human-Robot Interaction Design', 'State Estimation Methods', 'Simulation Environments for Robotics', 'Quadrupedal / Legged Robotics', 'Drone / UAV Control', 'Autonomous Driving Systems', 'Soft Robotics Materials & Control', 'Search Algorithms (BFS, DFS)', 'Informed Search (A*, IDA*)', 'Heuristic Design', 'Knowledge Representation Schemes (Semantic Nets, Frames)', 'Ontology Engineering (OWL, RDF)', 'Expert System Shells', 'Classical Planning (STRIPS)', 'Hierarchical Task Network (HTN)', 'Constraint Programming', 'SAT Solvers', 'Automated Theorem Proving', 'MLOps Principles & Practices', 'Model Deployment Patterns (API, Batch, Edge)', 'Containerization (Docker)', 'Orchestration (Kubernetes)', 'AWS SageMaker Suite', 'Azure Machine Learning', 'GCP AI Platform / Vertex AI', 'MLflow for Experiment Tracking', 'Kubeflow Pipelines', 'Data Pipeline Tools (Airflow, Prefect)', 'Apache Spark for Big Data ML', 'Distributed Training (Horovod)', 'Feature Stores (Feast, Tecton)', 'Model Monitoring & Alerting', 'CI/CD for ML Systems', 'Infrastructure as Code (Terraform)', 'Version Control for ML (DVC)', 'Quantum Machine Learning Algorithms', 'Evolutionary Computation', 'Genetic Algorithms', 'Swarm Intelligence (PSO, ACO)', 'Artificial Life', 'Multi-Agent Reinforcement Learning (MARL)', 'Computational Complexity', 'Information Geometry', 'Algorithmic Game Theory', 'Differential Privacy', 'Homomorphic Encryption for ML', 'Neurosymbolic AI', 'Cognitive Architectures', 'Bayesian Deep Learning', 'Probabilistic Programming (PyMC, Stan)', 'Gradient Checking', 'Hyperparameter Tuning (Grid Search, Random Search, Bayesian Opt.)', 'Learning Embeddings', 'Few-Shot Learning', 'Zero-Shot Learning', 'Self-Supervised Learning', 'Contrastive Learning', 'Active Learning', 'Online Learning', 'Bandit Algorithms', 'Recommendation Metrics', 'Natural Language Generation (NLG)', 'Semantic Parsing', 'Knowledge Graphs', 'Relational Databases for AI', 'Graph Databases (Neo4j)', '3D Object Recognition', 'Video Analysis', 'Event Cameras', 'Tactile Sensing', 'Force Control in Robotics', 'Whole-Body Control', 'Teleoperation', 'AI Safety & Alignment', 'Interpretable Machine Learning (LIME, SHAP)', 'Adversarial Attacks & Defense', 'Network Science', 'Bio-inspired Computing', 'Neuromorphic Computing', 'Computational Creativity', 'Digital Signal Processing (DSP)', 'Formal Methods in AI'
];

// --- DOM Elements ---
const canvas = document.getElementById('graphCanvas');
const ctx = canvas.getContext('2d');
const mainPageContainer = document.getElementById('main-page'); // Get the top bar container
const searchBar = document.getElementById('searchBar');
const searchButton = document.getElementById('searchButton');
const categoryButtonsContainer = document.getElementById('categoryButtons');
const topicContent = document.getElementById('topicContent');
const mainPage = document.getElementById('main-page'); // Keep reference if needed elsewhere
const topicPage = document.getElementById('topic-page');
const backButton = document.getElementById('backButton');


// --- Globals ---
let nodes = [];
let numNodes = potentialLabels.length; // Use all labels by default
let activeFilter = 'All';
let mouse = { x: null, y: null };
let centerPos = { x: 0, y: 0 }; // Edge anchor point
let devicePixelRatio = window.devicePixelRatio || 1;

// --- Physics/Interaction Parameters ---
const nodeTextSize = 8.5; // Updated text size
const edgeColor = 'rgba(100, 100, 100, 0.4)'; // Slightly fainter edges
const lineWidth = 0.5; // Thinner lines
const repulsionStrength = 0.9; // General mouse repulsion
const maxInteractionDist = 150;
const springStrength = 0.0025; // How strongly nodes return to target
const damping = 0.95; // Slows down movement

console.log(`Using ${potentialLabels.length} labels for nodes.`);

// --- Helper Functions ---
function getCategoryForLabel(label) {
    const lowerLabel = label.toLowerCase();
    for (const keyword in keywordToCategory) {
        let keywordPattern = keyword;
        // Regex for word boundaries, improved for specific cases
        if (keyword.length <= 3 || ['ai', 'cv', 'rl', 'go', 'r ', 'a*', 'svm', 'knn', 'pca', 'lda', 'gan', 'vae', 'cnn', 'rnn', 'gru', 'lstm', 'ner', 'ros', 'pid', 'tts', 'asr', 'eda', 'sql'].includes(keyword.trim().toLowerCase()) || !keyword.includes(' ')) {
            keywordPattern = `\\b${keyword.trim().replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')}\\b`;
        } else {
            keywordPattern = keyword.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
        }
        try {
            const regex = new RegExp(keywordPattern, 'i');
            if (regex.test(lowerLabel)) {
                const category = keywordToCategory[keyword];
                if (categoryColors[category]) { return category; }
                else { console.warn(`Category "${category}" for keyword "${keyword}" not in categoryColors.`); }
            }
        } catch (e) { console.error(`Regex error for keyword "${keyword}": ${e}`); }
    }
    return 'Default'; // Fallback category
}

function getColorForCategory(categoryName) {
    return categoryColors[categoryName] || defaultColor;
}

// Get the "no-go" zone around the top bar content (#main-page)
function getTopBarNoGoZone() {
    if (!mainPageContainer) return null; // Use the container reference
    try {
        const rect = mainPageContainer.getBoundingClientRect();
        if (!rect || rect.width <= 0 || rect.height <= 0) {
            // console.warn("Invalid top bar dimensions obtained."); // Reduce console noise
            return null;
        }
        // Use searchBarZonePadding to expand the zone
        return {
            top: rect.top - searchBarZonePadding, // Extend above
            left: rect.left - searchBarZonePadding, // Extend left
            bottom: rect.bottom + searchBarZonePadding, // Extend below
            right: rect.right + searchBarZonePadding, // Extend right
            width: rect.width + 2 * searchBarZonePadding,
            height: rect.height + 2 * searchBarZonePadding,
            centerX: rect.left + rect.width / 2,
            centerY: rect.top + rect.height / 2
        };
    } catch (e) {
        console.error("Error getting top bar bounds:", e);
        return null;
    }
}


// --- Node Class ---
// Uses the increased padding constants automatically
class Node {
    constructor(text) {
        this.text = text;
        this.category = getCategoryForLabel(this.text);
        this.color = getColorForCategory(this.category);
        this.x = 0; this.y = 0; this.originX = 0; this.originY = 0;
        this.vx = 0; this.vy = 0; this.targetX = 0; this.targetY = 0;
    }

    // Set initial random position within padded bounds, avoiding top bar zone
    setRandomOrigin(canvasWidthCSS, canvasHeightCSS) {
        const minX = sidePadding;
        const maxX = canvasWidthCSS - sidePadding;
        const minY = topPadding; // Use larger topPadding
        const maxY = canvasHeightCSS - bottomPadding; // Use larger bottomPadding
        let attempts = 0;
        const maxAttempts = 250; // Allow more attempts if area is crowded
        let tempX, tempY;
        let isInNoGoZone = false;
        const noGoZone = getTopBarNoGoZone(); // Check against top bar zone

        do {
            isInNoGoZone = false;
            if (maxX <= minX || maxY <= minY) {
                console.warn("Padded area invalid. Placing node center-ish.");
                tempX = canvasWidthCSS / 2;
                tempY = topPadding + (canvasHeightCSS - topPadding - bottomPadding) / 2;
                break;
            } else {
                tempX = minX + Math.random() * (maxX - minX);
                tempY = minY + Math.random() * (maxY - minY); // Within new padded bounds
            }

            // Check if initial random position is inside the top bar's no-go zone
            if (noGoZone &&
                tempX > noGoZone.left && tempX < noGoZone.right &&
                tempY > noGoZone.top && tempY < noGoZone.bottom) {
                isInNoGoZone = true;
            }
            attempts++;
        } while (isInNoGoZone && attempts < maxAttempts);

        // If still in zone after max attempts, push it out radially
        if (attempts >= maxAttempts && isInNoGoZone) {
             console.warn(`Max attempts (${maxAttempts}) reached placing node outside top bar zone. Pushing radially.`);
             let dxZone = tempX - noGoZone.centerX;
             let dyZone = tempY - noGoZone.centerY;
             let distZoneSq = dxZone * dxZone + dyZone * dyZone;
             if (distZoneSq < 0.01) { dxZone = 1; dyZone = 0; distZoneSq = 1;} // Avoid zero vector
             const distZone = Math.sqrt(distZoneSq);
             // Push just outside the padded zone radius
             const pushFactor = (Math.max(noGoZone.width, noGoZone.height) / 2 + 15) / distZone;
             tempX = noGoZone.centerX + dxZone * pushFactor;
             tempY = noGoZone.centerY + dyZone * pushFactor;
             // Re-clamp to main padded boundaries
             tempX = Math.max(minX, Math.min(maxX, tempX));
             tempY = Math.max(minY, Math.min(maxY, tempY));
        }

        this.originX = tempX; this.originY = tempY;
        this.x = this.originX; this.y = this.originY;
        this.targetX = this.originX; this.targetY = this.originY;
    }

    // Set new target position, adjusting if it falls inside the top bar zone
    setNewTarget(x, y) {
        const currentWidthCSS = parseInt(canvas.style.width, 10) || window.innerWidth;
        const currentHeightCSS = parseInt(canvas.style.height, 10) || window.innerHeight;
        const textMargin = nodeTextSize / 2; // Approx text bounds

        // Define allowed area using increased padding
        const minX = sidePadding + textMargin;
        const maxX = currentWidthCSS - sidePadding - textMargin;
        const minY = topPadding + textMargin;
        const maxY = currentHeightCSS - bottomPadding - textMargin;

        // Initial clamp to main boundaries
        let clampedX = Math.max(minX, Math.min(maxX, x));
        let clampedY = Math.max(minY, Math.min(maxY, y));

        // Check if the clamped target is inside the top bar's no-go zone
        const noGoZone = getTopBarNoGoZone();
        if (noGoZone &&
            clampedX > noGoZone.left && clampedX < noGoZone.right &&
            clampedY > noGoZone.top && clampedY < noGoZone.bottom)
        {
            // Push target radially outward from the zone center
            let dxZone = clampedX - noGoZone.centerX;
            let dyZone = clampedY - noGoZone.centerY;
            let distZoneSq = dxZone * dxZone + dyZone * dyZone;
            if (distZoneSq < 0.01) { // Avoid division by zero if exactly at center
                 dxZone = (Math.random() - 0.5) * 2; dyZone = (Math.random() - 0.5) * 2;
                 distZoneSq = dxZone * dxZone + dyZone * dyZone; if(distZoneSq < 0.01) distZoneSq = 1;
            }
            const distZone = Math.sqrt(distZoneSq);
            const nx = dxZone / distZone; // Normalized direction vector
            const ny = dyZone / distZone;
            // Estimate radius and push slightly beyond it
            const zoneRadius = Math.sqrt(noGoZone.width * noGoZone.width + noGoZone.height * noGoZone.height) / 2;
            const pushDistance = zoneRadius + 20; // Push a bit further out
            clampedX = noGoZone.centerX + nx * pushDistance;
            clampedY = noGoZone.centerY + ny * pushDistance;

            // Re-clamp to the main padded boundaries AFTER pushing
            clampedX = Math.max(minX, Math.min(maxX, clampedX));
            clampedY = Math.max(minY, Math.min(maxY, clampedY));
        }

        // Assign the final (potentially adjusted) target
        this.targetX = clampedX;
        this.targetY = clampedY;
        this.originX = this.targetX; // Update origin for stability
        this.originY = this.targetY;
    }

    // Update node physics and position
    update(canvasWidthCSS, canvasHeightCSS) {
        // --- Physics: Spring towards target ---
        let dxTarget = this.targetX - this.x;
        let dyTarget = this.targetY - this.y;
        this.vx += dxTarget * springStrength;
        this.vy += dyTarget * springStrength;

        // --- Physics: Mouse Repulsion ---
        if (mouse.x !== null && mouse.y !== null) {
             let dxMouse = this.x - mouse.x; let dyMouse = this.y - mouse.y;
             let distMouseSq = dxMouse * dxMouse + dyMouse * dyMouse;
             let maxInteractionDistSq = maxInteractionDist * maxInteractionDist;
             if (distMouseSq < maxInteractionDistSq && distMouseSq > 1) {
                 let distMouse = Math.sqrt(distMouseSq);
                 let force = repulsionStrength * (maxInteractionDist - distMouse) / (distMouse * maxInteractionDist);
                 this.vx += (dxMouse / distMouse) * force * 50; // Scaled force
                 this.vy += (dyMouse / distMouse) * force * 50;
             }
        }

        // --- Physics: Top Bar Zone Repulsion ---
        const noGoZone = getTopBarNoGoZone();
        if (noGoZone && this.x > noGoZone.left && this.x < noGoZone.right && this.y > noGoZone.top && this.y < noGoZone.bottom) {
            let dxZone = this.x - noGoZone.centerX;
            let dyZone = this.y - noGoZone.centerY;
            let distZoneSq = dxZone * dxZone + dyZone * dyZone;
            if (distZoneSq > 0.01) {
                let distZone = Math.sqrt(distZoneSq);
                // Apply stronger repulsion force radially outward
                let forceX = (dxZone / distZone) * searchBarZoneRepulsion;
                let forceY = (dyZone / distZone) * searchBarZoneRepulsion;
                this.vx += forceX;
                this.vy += forceY;
            } else { // If exactly at center, push randomly
                this.vx += (Math.random() - 0.5) * searchBarZoneRepulsion;
                this.vy += (Math.random() - 0.5) * searchBarZoneRepulsion;
            }
        }

        // --- Physics: Damping & Speed Limit ---
        this.vx *= damping;
        this.vy *= damping;
        const speedLimit = 20;
        const speedSq = this.vx * this.vx + this.vy * this.vy;
        if (speedSq > speedLimit * speedLimit) {
            const speed = Math.sqrt(speedSq);
            this.vx = (this.vx / speed) * speedLimit;
            this.vy = (this.vy / speed) * speedLimit;
        }
        if (Math.abs(this.vx) < 0.01) this.vx = 0;
        if (Math.abs(this.vy) < 0.01) this.vy = 0;

        // --- Update Position ---
        this.x += this.vx;
        this.y += this.vy;

        // --- Clamping (Screen Boundaries using INCREASED Padding) ---
        const textMargin = nodeTextSize / 2;
        const leftBoundary = sidePadding + textMargin;
        const rightBoundary = canvasWidthCSS - sidePadding - textMargin;
        const topBoundary = topPadding + textMargin; // Uses larger topPadding
        const bottomBoundary = canvasHeightCSS - bottomPadding - textMargin; // Uses larger bottomPadding

        // Bounce slightly off boundaries
        if (this.x < leftBoundary) { this.x = leftBoundary; this.vx *= -0.5; }
        if (this.x > rightBoundary) { this.x = rightBoundary; this.vx *= -0.5; }
        if (this.y < topBoundary) { this.y = topBoundary; this.vy *= -0.5; }
        if (this.y > bottomBoundary) { this.y = bottomBoundary; this.vy *= -0.5; }
    }

    // Draw node edge and text
    draw(ctx, centerPos) {
        ctx.beginPath();
        ctx.moveTo(this.x, this.y);
        ctx.lineTo(centerPos.x, centerPos.y); // Connect edge to anchor point
        ctx.strokeStyle = edgeColor;
        ctx.lineWidth = lineWidth * devicePixelRatio; // Scale line for DPI
        ctx.stroke();

        ctx.fillStyle = this.color;
        ctx.font = `${Math.round(nodeTextSize * devicePixelRatio)}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(this.text, this.x, this.y); // Draw text at CSS coords
    }
}

// --- Button Handling ---
function updateButtonStyles() {
    if (!categoryButtonsContainer) return;
    const buttons = categoryButtonsContainer.querySelectorAll('button');
    buttons.forEach(button => {
        button.classList.toggle('active', button.dataset.category === activeFilter);
    });
}

function handleCategoryClick(category) {
    console.log("Filtering by:", category);
    activeFilter = category;
    updateButtonStyles();
}

function createCategoryButtons() {
    if (!categoryButtonsContainer) {
        console.error("Category buttons container not found!");
        return;
    }
    categoryButtonsContainer.innerHTML = '';
    const allButton = document.createElement('button');
    allButton.textContent = 'All';
    allButton.dataset.category = 'All';
    allButton.onclick = () => handleCategoryClick('All');
    categoryButtonsContainer.appendChild(allButton);

    const categories = Object.keys(categoryColors);
    categories.sort((a, b) => { // Sort alphabetically, Default last
        if (a === 'Default') return 1; if (b === 'Default') return -1;
        return a.localeCompare(b);
    });

    categories.forEach(category => {
        const hasNodesInCategory = nodes.some(n => n.category === category);
        if ((category !== 'Default' && hasNodesInCategory) || (category === 'Default' && hasNodesInCategory)) {
            const button = document.createElement('button');
            button.textContent = category;
            const color = getColorForCategory(category);
            button.style.borderColor = color; // Add category color border
            button.dataset.category = category;
            button.onclick = () => handleCategoryClick(category);
            categoryButtonsContainer.appendChild(button);
        }
    });
    updateButtonStyles(); // Set initial 'All' as active
}

// --- Update Edge Anchor Position --- (MODIFIED)
function updateCenterPosition() {
    const currentWidthCSS = parseInt(canvas.style.width, 10) || window.innerWidth;
    const currentHeightCSS = parseInt(canvas.style.height, 10) || window.innerHeight;

    // Anchor X: Horizontally centered on screen
    centerPos.x = currentWidthCSS / 2;

    // Anchor Y: Vertically centered within the *allowed node area*
    // (between topPadding and bottomPadding)
    const nodeAreaHeight = Math.max(1, currentHeightCSS - topPadding - bottomPadding); // Avoid zero/negative height
    // Place anchor slightly above the middle of this area (e.g., 40-45% down)
    centerPos.y = topPadding + nodeAreaHeight * 0.42; // Tune this multiplier (0.0 to 1.0)

    // Clamp anchor point to be safely within the padded area
    const anchorMargin = 30; // Prevent anchor being exactly on padding line
    centerPos.x = Math.max(sidePadding + anchorMargin, Math.min(currentWidthCSS - sidePadding - anchorMargin, centerPos.x));
    centerPos.y = Math.max(topPadding + anchorMargin, Math.min(currentHeightCSS - bottomPadding - anchorMargin, centerPos.y));
}


// --- Event Handlers ---
function resizeCanvas() {
    devicePixelRatio = window.devicePixelRatio || 1;
    const currentWidthCSS = window.innerWidth;
    const currentHeightCSS = window.innerHeight;

    const oldWidthCSS = parseInt(canvas.style.width, 10) || 0;
    const oldHeightCSS = parseInt(canvas.style.height, 10) || 0;

    // Update canvas buffer size and CSS size
    canvas.width = Math.round(currentWidthCSS * devicePixelRatio);
    canvas.height = Math.round(currentHeightCSS * devicePixelRatio);
    canvas.style.width = `${currentWidthCSS}px`;
    canvas.style.height = `${currentHeightCSS}px`;

    updateCenterPosition(); // Recalculate edge anchor point

    // Reposition nodes relative to the new padded area size
    if (oldWidthCSS > 0 && oldHeightCSS > 0) {
        const oldMinX = sidePadding; const oldMaxX = oldWidthCSS - sidePadding;
        const oldMinY = topPadding; const oldMaxY = oldHeightCSS - bottomPadding;
        const oldRangeX = Math.max(1, oldMaxX - oldMinX);
        const oldRangeY = Math.max(1, oldMaxY - oldMinY);

        const newMinX = sidePadding; const newMaxX = currentWidthCSS - sidePadding;
        const newMinY = topPadding; const newMaxY = currentHeightCSS - bottomPadding;
        const newRangeX = Math.max(1, newMaxX - newMinX);
        const newRangeY = Math.max(1, newMaxY - newMinY);

        nodes.forEach(node => {
            const clampedOldOriginX = Math.max(oldMinX, Math.min(oldMaxX, node.originX));
            const clampedOldOriginY = Math.max(oldMinY, Math.min(oldMaxY, node.originY));
            const relativeX = (clampedOldOriginX - oldMinX) / oldRangeX;
            const relativeY = (clampedOldOriginY - oldMinY) / oldRangeY;
            let newTargetX = newMinX + relativeX * newRangeX;
            let newTargetY = newMinY + relativeY * newRangeY;
            // Use setNewTarget which handles top bar zone avoidance
            node.setNewTarget(newTargetX, newTargetY);
        });
         console.log("Canvas resized, nodes repositioned.");
    } else {
         console.warn("Resize with invalid old dimensions, resetting node origins.");
         nodes.forEach(node => node.setRandomOrigin(currentWidthCSS, currentHeightCSS));
    }
}

function updateMousePosition(event) {
    const rect = canvas.getBoundingClientRect();
    mouse.x = event.clientX - rect.left;
    mouse.y = event.clientY - rect.top;
}

// --- Animation Loop ---
function animate() {
    const currentWidthCSS = parseInt(canvas.style.width, 10);
    const currentHeightCSS = parseInt(canvas.style.height, 10);

    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas buffer
    ctx.save(); // Save context state
    ctx.scale(devicePixelRatio, devicePixelRatio); // Scale for high DPI

    // Update and draw visible nodes
    nodes.forEach(node => {
        node.update(currentWidthCSS, currentHeightCSS); // Update physics always
        if (activeFilter === 'All' || node.category === activeFilter) {
            node.draw(ctx, centerPos); // Draw if filter matches
        }
    });

    ctx.restore(); // Restore context state (removes scaling)
    requestAnimationFrame(animate); // Request next frame
}


// --- Initialization ---
function init() {
    devicePixelRatio = window.devicePixelRatio || 1;
    console.log("Device Pixel Ratio:", devicePixelRatio);
    const initialWidthCSS = window.innerWidth;
    const initialHeightCSS = window.innerHeight;

    canvas.width = Math.round(initialWidthCSS * devicePixelRatio);
    canvas.height = Math.round(initialHeightCSS * devicePixelRatio);
    canvas.style.width = `${initialWidthCSS}px`;
    canvas.style.height = `${initialHeightCSS}px`;

    updateCenterPosition(); // Set initial edge anchor

    // Create Nodes
    nodes = [];
    numNodes = potentialLabels.length; // Use all potential labels
    for (let i = 0; i < numNodes; i++) {
        const newNode = new Node(potentialLabels[i]);
        // setRandomOrigin now avoids top bar zone
        newNode.setRandomOrigin(initialWidthCSS, initialHeightCSS);
        nodes.push(newNode);
    }
    console.log(`Initialized ${nodes.length} nodes.`);

    createCategoryButtons(); // Generate filter buttons

    // Add Event Listeners
    window.addEventListener('resize', resizeCanvas);
    canvas.addEventListener('mousemove', updateMousePosition);
    canvas.addEventListener('mouseout', () => { mouse.x = null; mouse.y = null; });

    // Check required elements for topic display
    if (!topicContent || !mainPageContainer || !topicPage || !backButton || !searchButton || !searchBar || !categoryButtonsContainer) { // Added categoryButtonsContainer
         console.error("One or more essential DOM elements are missing!");
    } else {
         console.log("UI Elements Found.");
    }
}


// --- Topic Display Functionality --- (UPDATED FUNCTION)
function displayTopic(topicName) {
    const lowerTopicName = topicName.toLowerCase().trim();
    let foundTopicData = topics[lowerTopicName]; // Check pre-defined topics first

    if (!topicContent || !mainPageContainer || !topicPage || !categoryButtonsContainer) { // Added categoryButtonsContainer check
        console.error("Cannot display topic: Required page elements not found.");
        return;
    }

    let htmlContent = ''; // Initialize content string

    if (foundTopicData) {
        // --- Case 1: Found in the pre-defined topics object ---
        console.log(`Displaying pre-defined content for: ${topicName}`);
        // Use the original display logic for detailed topics
        // Check if category exists in the found data, otherwise use 'invert' or get it
        let topicCategory = foundTopicData.category || 'invert'; // Default if not specified
        if (topicCategory === 'invert' && topics[lowerTopicName]) { // If marked 'invert' but has entry, try to get real category
             topicCategory = getCategoryForLabel(lowerTopicName) || 'Default';
        } else if (!categoryColors[topicCategory]) { // If category invalid, fallback
            topicCategory = getCategoryForLabel(lowerTopicName) || 'Default';
        }

        htmlContent = `
            <h2>${topicName.charAt(0).toUpperCase() + topicName.slice(1)}</h2>
            <p style="margin-bottom: 15px;"><strong>Category:</strong> <span style="color: ${getColorForCategory(topicCategory)}; font-weight: bold;">${topicCategory}</span></p>
            <hr style="border-color: #444;">
            <h3>What is it?</h3> <p>${foundTopicData.what || 'N/A'}</p>
            <h3>Why is it important?</h3> <p>${foundTopicData.why || 'N/A'}</p>
            <h3>How does it work?</h3> <p>${foundTopicData.how || 'N/A'}</p>
            <h3>Example Use Cases:</h3> <ul>${(foundTopicData.use_cases || []).map(uc => `<li>${uc}</li>`).join('') || '<li>N/A</li>'}</ul>
            <h3>Analogy:</h3> <p>${foundTopicData.analogy || 'N/A'}</p>
        `;
    } else {
        // --- Case 2: Not in pre-defined topics, check if it's a valid node label ---
        // Find the label in potentialLabels (case-insensitive comparison)
        const potentialLabelMatch = potentialLabels.find(label => label.toLowerCase() === lowerTopicName);

        if (potentialLabelMatch) {
             // --- Case 2a: It's a valid node label, display generic info ---
             console.log(`Generating generic content for valid label: ${potentialLabelMatch}`);
             const category = getCategoryForLabel(potentialLabelMatch); // Get category using the matched label's original casing
             const displayTitle = potentialLabelMatch; // Use the correctly cased label from the array

             htmlContent = `
                <h2>${displayTitle}</h2>
                <p style="margin-bottom: 15px;"><strong>Category:</strong> <span style="color: ${getColorForCategory(category)}; font-weight: bold;">${category || 'Default'}</span></p>
                <hr style="border-color: #444;">
                <h3>Overview</h3>
                <p>This topic, "<strong>${displayTitle}</strong>", is a concept within the broader category of <strong>${category || 'Default'}</strong> in the field of AI and Data Science.</p>
                <p><em>Detailed 'What, Why, How' information specific to only "${displayTitle}" is not elaborated in this demo. However, understanding its role within the <strong>${category || 'Default'}</strong> category provides context.</em></p>
                <h3>General Points (related to ${category || 'Default'}):</h3>
                <ul>
                    <li>Often used techniques or concepts within the ${category || 'Default'} workflow.</li>
                    <li>Understanding this helps in comprehending related topics in the same category.</li>
                    <li>May involve specific algorithms, data structures, or theoretical principles relevant to ${category || 'Default'}.</li>
                </ul>
                 <p>Explore the graph or use the '<strong>${category || 'Default'}</strong>' filter button to see related concepts.</p>
                `;

        } else {
            // --- Case 3: Topic not found in pre-defined list OR potentialLabels ---
            console.log(`Topic not found or not a node label: ${topicName}`);
            htmlContent = `
                <h2>Topic Not Found</h2>
                <p>Sorry, "<strong>${topicName}</strong>" is not recognized as a specific topic with detailed information or a node label in this visualization.</p>
                <p>Please try searching for a term visible in the graph, or explore using the category filters.</p>
                <hr style="border-color: #444; margin: 15px 0;">
                <p>Key topics with full details available:</p>
                <ul>
                    <li>Data Science</li>
                    <li>Machine Learning</li>
                    <li>Artificial Intelligence</li>
                </ul>
            `;
        }
    }

    // Update the DOM to show the content and hide the main graph view
    topicContent.innerHTML = htmlContent;
    mainPageContainer.classList.add('hidden'); // Hide top bar container
    categoryButtonsContainer.classList.add('hidden'); // Also hide category buttons
    topicPage.classList.remove('hidden'); // Show topic page
    topicPage.scrollTop = 0; // Scroll to the top of the topic page
}


// Add Listeners for Search and Back Button
if (searchBar && searchButton) {
    searchBar.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const topicName = searchBar.value.trim();
            if (topicName) displayTopic(topicName);
        }
    });
    searchButton.addEventListener('click', () => {
        const topicName = searchBar.value.trim();
        if (topicName) displayTopic(topicName);
    });
} else { console.warn("Search input or button element not found."); }

if (backButton && mainPageContainer && topicPage && categoryButtonsContainer) {
    backButton.addEventListener('click', () => {
        mainPageContainer.classList.remove('hidden'); // Show top bar
        categoryButtonsContainer.classList.remove('hidden'); // Show category buttons
        topicPage.classList.add('hidden'); // Hide topic page
        if(searchBar) searchBar.value = ''; // Clear search bar
    });
} else { console.warn("Back button or essential page elements for navigation not found."); }


// --- Start ---
if (!canvas || !ctx) {
    console.error("ERROR: Canvas element or 2D context not found!");
    alert("ERROR: Cannot initialize graph - Canvas setup failed.");
} else if (!potentialLabels || potentialLabels.length === 0) {
     console.error("ERROR: The potentialLabels array is empty!");
     alert("ERROR: Label list is missing. Cannot initialize graph.");
} else {
     // Ensure DOM is fully loaded before initializing
     if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            console.log("DOM ready, initializing...");
            init();
            animate(); // Start animation loop
        });
     } else {
        console.log("DOM already ready, initializing immediately...");
        init();
        animate(); // Start animation loop
     }
}