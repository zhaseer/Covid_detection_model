# Use a minimal base image with Python 3.9 installed
FROM python:3.9

# Set the working directory inside the container to /app
WORKDIR /app

# Copy requirement files first for better Docker caching
COPY requirements.txt .

# Install Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- CRITICAL FIX: Explicitly set Keras backend ---
ENV KERAS_BACKEND=tensorflow

# Copy application files and the model file (must be named tuned_ai_model_best_lat.keras)
COPY app.py .
COPY tuned_ai_model_best_lat.keras .

# Recommended security practice
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Define the command to run the Streamlit app, re-including the CRITICAL security flag
# This flag should prevent the 403 Forbidden error caused by XSRF protection in proxy environments.
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
