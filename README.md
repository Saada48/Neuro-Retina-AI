# Neuro-Retina-AI
The Neuro Retina system is a clinical-grade, deep learning application designed to accurately detect and classify 8 distinct retinal pathologies from medical scans.  By bridging the gap between advanced artificial intelligence and healthcare diagnostics, it provides an accessible, reliable, and end-to-end screening tool.
# Key Features
Deep Learning Architecture: Powered by the highly efficient MobileNetV3 model, ensuring rapid and accurate classification of complex retinal images without requiring massive computational overhead.

Explainable AI (XAI): Integrates Grad-CAM technology to generate visual heatmaps.  This provides critical clinical transparency by showing exactly which regions of the eye influenced the model's diagnosis.

Robust Image Validation: Incorporates a smart filtration step that automatically detects and rejects non-retinal or invalid images before processing, preventing false reads.

Comprehensive UI: Features a clean, interactive Streamlit dashboard tailored for ease of use, allowing seamless image uploads and instant diagnostic results.

Patient Management: Utilizes a lightweight SQLite database acting as a secure patient registry to locally track medical records and historical diagnostic data.

Smart Medical Assistant: Includes a context-aware chatbot grounded in a specialized medical knowledge base to answer disease-specific questions and provide diagnostic context directly within the app.
